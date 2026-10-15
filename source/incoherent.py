"""
Некогерентная схема умножения в домене интенсивностей (самая дешёвая установка).

Установка (оси как в основном симуляторе: X — ось вектора / свёртки, Y — ось выходов):

    линейка светодиодов (или строка экрана)        z = 0
      | скрещенная линза L1, z = d:
      |   по X — изображение 2f–2f (светодиод j -> столбец j маски),
      |   по Y — Фурье (каждая точка светодиода даёт наклонную плоскую волну,
      |          свет «растягивается» на все строки маски)
    маска W (плёнка или монохромный ЖК),            z = 2d
      | скрещенная линза L3, z = 3d:
      |   по Y — изображение 2f–2f (строка i -> фотодетектор i),
      |   по X — сбор всего света строки на детектор
    линейка фотодетекторов                          z = 4d

Чем это отличается от когерентного OpticalMul:
  * значения кодируются ИНТЕНСИВНОСТЬЮ: x — мощность светодиода, W — пропускание
    маски по интенсивности (амплитуда sqrt(W)), детектор читает мощность.
    Корня на выходе нет, результат линеен по x;
  * источник пространственно некогерентный: каждая точка светодиода распространяется
    отдельно, на детекторе складываются интенсивности;
  * по X детектор просто собирает мощность строки. В когерентной схеме L3 делает
    Фурье, и камера читает нулевую частоту (когерентную сумму). Здесь интерферировать
    нечему: при идеальном сборе мощность не зависит от фаз (равенство Парсеваля),
    поэтому выход по X = сумма |поле|^2 по столбцам маски;
  * по Y свет от одной точки светодиода освещает все строки маски когерентно
    (это плоская волна), поэтому изображение строк на детекторы — частично
    когерентное: степень когерентности задаёт высота светодиода src_height.
    Из-за этого отклик слабо нелинеен по W (интерференция соседних строк).
    Это физика схемы, а не артефакт.

Распространение в свободном пространстве и линзы — те же формулы, что в
propagator.py (sinc-базис, ядро Френеля, тонкие линзы). Фазы линз применяются
поэлементно, без плотных диагональных матриц, чтобы влезали большие апертуры.

Здесь только идеальная оптика. Шум детектора, АЦП, гамма маски, дрейф —
следующий этап (снаружи, поверх сырого выхода forward).

Правила сетки (results/sim_limits, 14.10.2026): шаг δ = pitch/split ≤ λ/(8·NA),
шаг линзы ≤ λ/(2·(NA + ширина_поля/(2d))). check_grid() предупреждает о нарушении.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence, Tuple

import torch as _torch
import torch.nn as _nn

from .config import ConfigOpticBase as _ConfigOpticBase, ConfigDesignPlane as _Plane
from .propagator import PropagatorSinc as _PropSinc


class IncoherentConfig:
    """
    Параметры некогерентной установки. Длины в метрах.

    Args:
        n_in:           длина вектора (число светодиодов = столбцов маски).
        n_out:          число выходов (строк маски = фотодетекторов).
        pitch_x:        шаг элемента вдоль вектора (светодиоды, столбцы маски).
        pitch_y:        шаг строк маски и детекторов (по умолчанию pitch_x).
        split_x/split_y: подотсчёты на элемент (численная сетка, см. check_grid).
        src_fill:       доля шага pitch_x, которая светится у светодиода.
        src_height:     высота светящейся области светодиода по Y.
        src_samples_y:  число некогерентных точек по высоте светодиода.
        mask_fill:      открытая доля ячейки маски по каждой оси (зазор между ячейками).
        det_fill:       доля шага pitch_y, которую занимает фотодетектор.
        wavelength:     длина волны (центральная).
        spectrum:       [(λ, вес), ...] для широкополосного светодиода; None — одна λ.
        distance:       d, расстояние между плоскостями (f = d у Фурье-линз, d/2 у изображающих).
        lens_pixel_size, lens_size: сетка линз; апертура D = lens_size·lens_pixel_size.
    """

    def __init__(self, n_in: int, n_out: int, pitch_x: float,
                 pitch_y: Optional[float] = None,
                 split_x: int = 8, split_y: int = 8,
                 src_fill: float = 1.0, src_height: Optional[float] = None,
                 src_samples_y: int = 8,
                 mask_fill: float = 1.0, det_fill: float = 1.0,
                 wavelength: float = 532e-9,
                 spectrum: Optional[Sequence[Tuple[float, float]]] = None,
                 distance: float = 0.1,
                 lens_pixel_size: float = 1.8e-6, lens_size: int = 8192):
        self.n_in, self.n_out = n_in, n_out
        self.pitch_x = pitch_x
        self.pitch_y = pitch_x if pitch_y is None else pitch_y
        self.split_x, self.split_y = split_x, split_y
        self.src_fill = src_fill
        self.src_height = self.pitch_y if src_height is None else src_height
        self.src_samples_y = src_samples_y
        self.mask_fill, self.det_fill = mask_fill, det_fill
        self.wavelength = wavelength
        self.spectrum = list(spectrum) if spectrum else [(wavelength, 1.0)]
        self.distance = distance
        self.lens_pixel_size, self.lens_size = lens_pixel_size, lens_size

    @property
    def aperture(self) -> float:
        return self.lens_size * self.lens_pixel_size

    @property
    def na(self) -> float:
        """NA изображающих осей (линза на расстоянии d от предмета)."""
        return self.aperture / 2 / self.distance

    def planes(self):
        """Расчётные плоскости в формате основного симулятора: (Y, X)."""
        src = _Plane((self.src_samples_y, self.n_in * self.split_x),
                     aperture=(self.src_height, self.n_in * self.pitch_x))
        lens = _Plane(self.lens_size, self.lens_pixel_size)
        mask = _Plane((self.n_out * self.split_y, self.n_in * self.split_x),
                      aperture=(self.n_out * self.pitch_y, self.n_in * self.pitch_x))
        det = _Plane((self.n_out * self.split_y, 1),
                     aperture=(self.n_out * self.pitch_y, self.pitch_x))
        return src, lens, mask, det


def _sinc_op(difference, pixel_in, pixel_out, opt):
    """Оператор распространения по одной оси из PropagatorSinc (формула оригинала, без изменений)."""
    return _PropSinc._PropagatorSinc__get_operator_for_dim(None, pixel_in, pixel_out, difference, opt)


def _lit(split: int, fill: float) -> _torch.Tensor:
    """Маска подотсчётов элемента: светятся/открыты центральные round(fill·split)."""
    n = max(1, min(split, int(round(fill * split))))
    m = _torch.zeros(split)
    a = (split - n) // 2
    m[a:a + n] = 1
    return m


class IncoherentOpticalMul(_nn.Module):
    """
    Некогерентное умножение в домене интенсивностей. Контракт как у OpticalMul:
        input (B, C, H, W) — интенсивности светодиодов, >= 0;
        other (B, C, W, K) — пропускание маски по интенсивности, [0, 1];
        выход (B, C, H, K) — мощность на фотодетекторах (линейна по input).
    Масштаб (потери, апертура) снимается калибровкой, как и в железе.
    """

    def __init__(self, config: IncoherentConfig, device=None, chunk: int = 1024):
        """
        device — где собирать операторы (на GPU быстрее; буферы потом можно перенести .to()).
        chunk  — сколько отсчётов линзы обрабатывать за раз: операторы свободного
                 пространства строятся по частям и сразу сворачиваются через линзу,
                 поэтому плотные матрицы «плоскость × линза» целиком не хранятся.
        """
        super().__init__()
        self.config = c = config
        src, lens, mask, det = c.planes()
        dev = _torch.device(device or 'cpu')
        dbl = _torch.float64
        xs, ys = src.linspace_by_x.to(dbl), src.linspace_by_y.to(dbl)
        xm, ym = mask.linspace_by_x.to(dbl), mask.linspace_by_y.to(dbl)
        yo = det.linspace_by_y.to(dbl)
        xl = lens.linspace_by_x.to(dbl)
        lp = c.lens_pixel_size
        Kx, G = [], []
        for lam, w in c.spectrum:
            opt = _ConfigOpticBase(lam, c.distance)
            k = opt.K

            def op(x_in, pin, x_out, pout):
                """Оператор sinc-базиса (out, in) — та же формула, что в PropagatorSinc."""
                return _sinc_op(x_in[None, :] - x_out[:, None], pin, pout, opt).to(dev, _torch.complex64)

            hx1 = _torch.zeros(len(xm), len(xs), dtype=_torch.complex64, device=dev)
            hy1 = _torch.zeros(len(ym), len(ys), dtype=_torch.complex64, device=dev)
            hy2 = _torch.zeros(len(yo), len(ym), dtype=_torch.complex64, device=dev)
            for i in range(0, len(xl), chunk):
                l = xl[i:i + chunk]
                # Фазы скрещенных линз — те же, что в PropagatorCrossLens:
                # L1: X -> exp(-ik x²/d) (изображение), Y -> exp(-ik y²/2d) (Фурье);
                # L3 (транспонированная): Y -> exp(-ik y²/d) (изображение).
                l1x = _torch.exp(-1j * k / c.distance * l ** 2).to(dev, _torch.complex64)
                l1y = _torch.exp(-1j * k / 2 / c.distance * l ** 2).to(dev, _torch.complex64)
                l3y = l1x
                # X: источник -> маска. Фаза полевой линзы у маски на |·|² не влияет.
                hx1 += op(l, lp, xm, mask.pixel_size_by_x) @ (
                    l1x[:, None] * op(xs, src.pixel_size_by_x, l, lp))
                # Y: точка источника -> строки маски; строки маски -> детекторы
                hy1 += op(l, lp, ym, mask.pixel_size_by_y) @ (
                    l1y[:, None] * op(ys, src.pixel_size_by_y, l, lp))
                hy2 += op(l, lp, yo, det.pixel_size_by_y) @ (
                    l3y[:, None] * op(ym, mask.pixel_size_by_y, l, lp))
            Kx.append(w * hx1.abs().square().T.contiguous())       # (Nxs, Nxm)
            # G[ys] = hy2 · diag(hy1[:, ys]): поле на детекторе от одной точки светодиода
            g = hy2[None, :, :] * hy1.T[:, None, :]                 # (Nys, Nyo, Nym)
            G.append(g.reshape(-1, g.shape[-1]))
        self.register_buffer('_kx', _torch.stack(Kx), persistent=False)    # (L, Nxs, Nxm)
        self.register_buffer('_g', _torch.stack(G), persistent=False)      # (L, Nys·Nyo, Nym)
        self.register_buffer('_src_lit', _lit(c.split_x, c.src_fill).to(dev), persistent=False)
        mm = _torch.outer(_lit(c.split_y, c.mask_fill), _lit(c.split_x, c.mask_fill))
        self.register_buffer('_mask_lit', mm.to(dev), persistent=False)
        self.register_buffer('_det_lit', _lit(c.split_y, c.det_fill).to(dev), persistent=False)

    # --- этапы схемы ---------------------------------------------------------
    def _crop(self, w: int, k: int):
        """
        Срезы буферов под матрицу w×k в центре поля. Вне неё светодиоды выключены,
        а ячейки маски закрыты, поэтому туда не приходит и оттуда не уходит свет:
        срез даёт тот же результат, что и расчёт на полном поле, но стоит
        пропорционально размеру матрицы, а не поля.
        """
        c = self.config
        sx, sy = c.split_x, c.split_y
        pw, pk = (c.n_in - w) // 2, (c.n_out - k) // 2
        xm = slice(pw * sx, (pw + w) * sx)                                  # столбцы маски
        xs = slice((c.n_in - pw - w) * sx, (c.n_in - pw) * sx)              # светодиоды (перевёрнуты)
        ym = slice(pk * sy, (pk + k) * sy)                                  # строки маски
        yo = slice((c.n_out - pk - k) * sy, (c.n_out - pk) * sy)            # детекторы (перевёрнуты)
        return xs, xm, ym, yo

    def illuminate(self, input: _torch.Tensor, crop=None) -> _torch.Tensor:
        """Интенсивность на столбцах маски по X: (…, H, Nxm) для каждой λ -> (L, …, H, Nxm)."""
        x = input.float().flip(-1)                          # изображение 2f–2f переворачивает
        x = _torch.kron(x.contiguous(), self._src_lit[None, :])
        kx = self._kx if crop is None else self._kx[:, crop[0], crop[1]]
        return _torch.einsum('...hs,lsm->l...hm', x, kx)

    def row_response(self, other: _torch.Tensor, crop=None,
                     max_elems: int = 2 ** 28) -> _torch.Tensor:
        """
        R[l, …, K, Nxm]: мощность на детекторе k от единичной засветки столбца xm.
        Матрицы батча обрабатываются частями так, чтобы промежуточное поле на
        детекторах занимало не больше max_elems комплексных чисел (по умолчанию 2 ГБ).
        """
        c = self.config
        k = other.shape[-1]
        n_ys = c.src_samples_y
        g = self._g.unflatten(1, (n_ys, -1))                    # (L, Nys, Nyo, Nym)
        if crop is not None:
            g = g[:, :, crop[3], crop[2]]
        n_yo = g.shape[2]
        batch = other.shape[:-2]
        flat = other.reshape(-1, *other.shape[-2:])
        n_xm = other.shape[-2] * c.split_x
        step = max(1, max_elems // (n_ys * n_yo * n_xm))
        out = []
        for l in range(g.shape[0]):
            gl = g[l].reshape(-1, g.shape[-1])
            parts = []
            for i in range(0, flat.shape[0], step):
                t = flat[i:i + step].float().clamp(0., 1.).clamp_min(1e-12).sqrt().transpose(-2, -1)
                t = _torch.kron(t.contiguous(), self._mask_lit).to(_torch.complex64)  # (b, Nym, Nxm)
                a = gl @ t                                      # (b, Nys·Nyo, Nxm)
                p = a.abs().square().unflatten(-2, (n_ys, n_yo)).sum(-3) / n_ys
                p = p.unflatten(-2, (k, c.split_y))             # (b, K, split_y, Nxm)
                p = (p * self._det_lit[:, None]).sum(-2)
                parts.append(p.flip(-2))                        # изображение переворачивает строки
            out.append(_torch.cat(parts).reshape(*batch, k, n_xm))
        return _torch.stack(out)

    def response(self, other: _torch.Tensor) -> _torch.Tensor:
        """R для матрицы other (её можно посчитать один раз для статических весов)."""
        return self.row_response(other, self._crop(*other.shape[-2:]))

    def forward(self, input: _torch.Tensor, other: _torch.Tensor,
                response: Optional[_torch.Tensor] = None) -> _torch.Tensor:
        c = self.config
        w, k = other.shape[-2], other.shape[-1]
        assert w <= c.n_in and k <= c.n_out, "матрица больше поля установки"
        # Меньшая матрица стоит в центре поля (как центрирование в OpticalMul).
        crop = self._crop(w, k)
        ix = self.illuminate(input, crop)                       # (L, …, H, w·sx)
        r = self.row_response(other, crop) if response is None else response
        return (ix @ r.transpose(-2, -1)).sum(0)

    # --- диагностика ---------------------------------------------------------
    @_torch.no_grad()
    def illumination_profile(self) -> dict:
        """
        Равномерность засветки маски (тот же вопрос, что в аналитике по BERT):
          rows — относительная засветка строк маски (растяжка по Y, от центра = 1);
          cols — относительная засветка столбцов (изображение светодиодов по X).
        """
        c = self.config
        ones_x = _torch.ones(1, c.n_in)
        cols = self.illuminate(ones_x).sum(0)[0]
        cols = cols.unflatten(-1, (c.n_in, c.split_x)).sum(-1)
        r = self.row_response(_torch.ones(c.n_in, c.n_out)).sum(0)   # (K, Nxm)
        rows = r.sum(-1)
        norm = lambda v: v / v[len(v) // 2]
        return dict(rows=norm(rows), cols=norm(cols))

    @_torch.no_grad()
    def flat_field(self) -> _torch.Tensor:
        """Отклик каждого детектора на единичный вектор и открытую маску (калибровка по столбцам)."""
        c = self.config
        return self(_torch.ones(1, 1, 1, c.n_in), _torch.ones(1, 1, c.n_in, c.n_out))[0, 0, 0]

    def check_grid(self, warn: bool = True) -> dict:
        """Проверка численной сетки по правилам results/sim_limits (δ ≤ λ/8NA и т. д.)."""
        c = self.config
        lam = min(l for l, _ in c.spectrum)
        na = c.na
        d_max = lam / (8 * na)
        field = max(c.n_in * c.pitch_x, c.n_out * c.pitch_y)
        lp_max = lam / (2 * (na + field / 2 / c.distance))
        res = dict(na=na, delta_x=c.pitch_x / c.split_x, delta_y=c.pitch_y / c.split_y,
                   delta_max=d_max, lens_pix=c.lens_pixel_size, lens_pix_max=lp_max,
                   u_x=c.pitch_x * na / lam, u_y=c.pitch_y * na / lam)
        msgs = []
        if max(res['delta_x'], res['delta_y']) > d_max:
            msgs.append(f"шаг сетки {max(res['delta_x'], res['delta_y'])*1e6:.2f} мкм > "
                        f"λ/(8NA) = {d_max*1e6:.2f} мкм: помехи пикселя занижены")
        if c.lens_pixel_size > lp_max:
            msgs.append(f"шаг линзы {c.lens_pixel_size*1e6:.2f} мкм > {lp_max*1e6:.2f} мкм: "
                        f"край поля считается неверно")
        if field > c.aperture:
            msgs.append(f"поле {field*1e3:.2f} мм шире линзы {c.aperture*1e3:.2f} мм")
        if warn:
            for m in msgs:
                warnings.warn(m)
        res['ok'] = not msgs
        return res
