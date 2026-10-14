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

    def __init__(self, config: IncoherentConfig):
        super().__init__()
        self.config = c = config
        src, lens, mask, det = c.planes()
        Kx, G = [], []
        for lam, w in c.spectrum:
            opt = _ConfigOpticBase(lam, c.distance)
            k = opt.K
            xl, yl = lens.linspace_by_x.double(), lens.linspace_by_y.double()
            # Фазы скрещенных линз — те же, что в PropagatorCrossLens:
            # L1: X -> exp(-ik x²/d) (изображение), Y -> exp(-ik y²/2d) (Фурье);
            # L3 (транспонированная): Y -> exp(-ik y²/d) (изображение).
            l1x = _torch.exp(-1j * k / c.distance * xl ** 2)
            l1y = _torch.exp(-1j * k / 2 / c.distance * yl ** 2)
            l3y = _torch.exp(-1j * k / c.distance * yl ** 2)
            s_sl = _PropSinc(src, lens, opt)
            s_lm = _PropSinc(lens, mask, opt)
            s_ml = _PropSinc(mask, lens, opt)
            s_ld = _PropSinc(lens, det, opt)
            # X: источник -> маска (Nxs, Nxm). Фаза полевой линзы у маски на |·|² не влияет.
            hx1 = (s_sl.operator_X * l1x[None, :].to(s_sl.operator_X.dtype)) @ s_lm.operator_X
            # Y: точка источника -> строки маски (Nym, Nys); строки маски -> детектор (Nyo, Nym)
            hy1 = s_lm.operator_Y @ (l1y[:, None].to(s_sl.operator_Y.dtype) * s_sl.operator_Y)
            hy2 = s_ld.operator_Y @ (l3y[:, None].to(s_ml.operator_Y.dtype) * s_ml.operator_Y)
            Kx.append(w * hx1.abs().square().float())
            # G[ys] = hy2 · diag(hy1[:, ys]): поле на детекторе от одной точки светодиода
            g = hy2[None, :, :] * hy1.T[:, None, :]            # (Nys, Nyo, Nym)
            G.append(g.reshape(-1, g.shape[-1]).to(_torch.complex64))
        self.register_buffer('_kx', _torch.stack(Kx), persistent=False)    # (L, Nxs, Nxm)
        self.register_buffer('_g', _torch.stack(G), persistent=False)      # (L, Nys·Nyo, Nym)
        self.register_buffer('_src_lit', _lit(c.split_x, c.src_fill), persistent=False)
        mm = _torch.outer(_lit(c.split_y, c.mask_fill), _lit(c.split_x, c.mask_fill))
        self.register_buffer('_mask_lit', mm, persistent=False)
        self.register_buffer('_det_lit', _lit(c.split_y, c.det_fill), persistent=False)

    # --- этапы схемы ---------------------------------------------------------
    def illuminate(self, input: _torch.Tensor) -> _torch.Tensor:
        """Интенсивность на столбцах маски по X: (…, H, Nxm) для каждой λ -> (L, …, H, Nxm)."""
        x = input.float().flip(-1)                          # изображение 2f–2f переворачивает
        x = _torch.kron(x.contiguous(), self._src_lit[None, :])
        return _torch.einsum('...hs,lsm->l...hm', x, self._kx)

    def row_response(self, other: _torch.Tensor) -> _torch.Tensor:
        """R[l, …, K, Nxm]: мощность на детекторе k от единичной засветки столбца xm."""
        c = self.config
        t = other.float().clamp(0., 1.).clamp_min(1e-12).sqrt().transpose(-2, -1)  # (…, K, W)
        t = _torch.kron(t.contiguous(), self._mask_lit).to(_torch.complex64)        # (…, Nym, Nxm)
        n_ys, n_yo = c.src_samples_y, c.n_out * c.split_y
        out = []
        for l in range(self._g.shape[0]):
            a = self._g[l] @ t                                  # (…, Nys·Nyo, Nxm)
            p = a.abs().square().unflatten(-2, (n_ys, n_yo)).sum(-3) / n_ys
            p = p.unflatten(-2, (c.n_out, c.split_y))          # (…, K, split_y, Nxm)
            p = (p * self._det_lit[:, None]).sum(-2)
            out.append(p.flip(-2))                              # изображение переворачивает строки
        return _torch.stack(out)

    def forward(self, input: _torch.Tensor, other: _torch.Tensor) -> _torch.Tensor:
        c = self.config
        w, k = other.shape[-2], other.shape[-1]
        assert w <= c.n_in and k <= c.n_out, "матрица больше поля установки"
        # Меньшая матрица занимает центр поля: лишние светодиоды выключены,
        # лишние ячейки маски закрыты (как центрирование в OpticalMul).
        pw, pk = (c.n_in - w) // 2, (c.n_out - k) // 2
        if w < c.n_in or k < c.n_out:
            input = _torch.nn.functional.pad(input, (pw, c.n_in - w - pw))
            other = _torch.nn.functional.pad(other, (pk, c.n_out - k - pk, pw, c.n_in - w - pw))
        ix = self.illuminate(input)                             # (L, …, H, Nxm)
        r = self.row_response(other)                            # (L, …, K, Nxm)
        return (ix @ r.transpose(-2, -1)).sum(0)[..., pk:pk + k]

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
