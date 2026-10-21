"""
Учебная версия IncoherentOpticalMul, написанная по образцу optical_mul.py.

Физика та же, что в incoherent.py, но поле честно проходит через цепочки
пропагаторов оригинала (PropagatorSinc + PropagatorCrossLens), как в OpticalMul.
Разница с когерентной схемой видна в forward:
  * в OpticalMul весь вектор — одно когерентное поле, пропускается один раз;
  * здесь источник некогерентный: КАЖДАЯ точка светодиода пропускается отдельно,
    на детекторе складываются интенсивности |поле|², а не поля.

Код медленный (цикл по точкам источника, плотные матрицы линз) и нужен только
для понимания и сверки на маленьких матрицах. Для расчётов — incoherent.py.
"""
import torch as _torch
import torch.nn as _nn

from .config import ConfigOpticBase as _ConfigOpticBase
from .propagator import PropagatorCrossLens as _PropCrossLens, PropagatorSinc as _PropSinc, Propagator as _Prop
from .incoherent import IncoherentConfig as _IncoherentConfig, _lit


class IncoherentOpticalMulNaive(_nn.Module):
    def __init__(self, config: _IncoherentConfig):
        super().__init__()
        self.config = c = config
        assert len(c.spectrum) == 1, "учебная версия: одна длина волны"
        src, lens, mask, det = c.planes()
        opt = _ConfigOpticBase(c.wavelength, c.distance)

        # Путь 1, как _propagator_one в OpticalMul: светодиоды -> L1 -> маска.
        # Скрещенная линза та же: по X изображение 2f-2f, по Y Фурье (растяжка на все строки).
        # Цилиндрической линзы у маски нет: её фаза по X не влияет на мощность.
        self._propagator_one: _Prop = (_PropSinc(src, lens, opt)
                                       + _PropCrossLens(lens, opt)
                                       + _PropSinc(lens, mask, opt))
        # Путь 2, как _propagator_two: маска -> L3 (транспонированная) -> детекторы.
        # Используем только его ось Y (изображение строки i на детектор i).
        # Ось X не распространяем: детектор собирает ВСЮ мощность строки, а по
        # равенству Парсеваля она равна сумме |поле|² по столбцам прямо за маской.
        self._propagator_two: _Prop = (_PropSinc(mask, lens, opt)
                                       + _PropCrossLens(lens, opt).T
                                       + _PropSinc(lens, det, opt))

        # Аналоги _kron_vec_utils / _kron_mat_utils, но с учётом заполнения элемента.
        self.register_buffer('_src_lit', _lit(c.split_x, c.src_fill))
        self.register_buffer('_mask_lit', _torch.outer(_lit(c.split_y, c.mask_fill),
                                                       _lit(c.split_x, c.mask_fill)))
        self.register_buffer('_det_lit', _lit(c.split_y, c.det_fill))

    def prepare_vector(self, data):
        """Интенсивности светодиодов по подотсчётам X: (…, H, W·split_x). Поля здесь нет."""
        data = data.float().flip(-1)                       # изображение переворачивает
        return _torch.kron(data.contiguous(), self._src_lit[None, :])

    def prepare_matrix(self, data):
        """Амплитудное пропускание маски: sqrt(W), т.к. W задано по интенсивности. (…, Nym, Nxm)."""
        data = data.float().clamp(0., 1.).sqrt().transpose(-2, -1)
        return _torch.kron(data.contiguous(), self._mask_lit).cfloat()

    def prepare_out(self, power):
        """Мощность на подотсчётах детектора (…, Nyo) -> по детекторам (…, K)."""
        c = self.config
        power = power.unflatten(-1, (c.n_out, c.split_y))
        return (power * self._det_lit).sum(-1).flip(-1)

    def forward(self, input, other):
        c = self.config
        x = self.prepare_vector(input)                     # (…, H, Nxs) — мощности, >= 0
        t = self.prepare_matrix(other)                     # (…, Nym, Nxm)
        n_ys, n_xs = c.src_samples_y, x.shape[-1]
        out = 0
        # Некогерентный источник = набор независимых точек. Каждая точка —
        # отдельное когерентное поле; их интенсивности на детекторе складываются.
        for ys in range(n_ys):
            for xs in range(n_xs):
                point = _torch.zeros(n_ys, n_xs, dtype=_torch.cfloat)
                point[ys, xs] = 1.0
                field = self._propagator_one(point, None)              # поле на маске (Nym, Nxm)
                field = field * t                                      # маска
                field = self._propagator_two.operator_Y @ field        # Y: строки -> детекторы (…, Nyo, Nxm)
                power = field.abs().square().sum(-1)                   # X: детектор собирает всю мощность
                # Вклад линеен по мощности светодиода: x[h, xs] · power[k]
                out = out + x[..., :, xs, None] * power[..., None, :] / n_ys
        return self.prepare_out(out)
