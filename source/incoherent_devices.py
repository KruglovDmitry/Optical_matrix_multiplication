"""
Слой реальных устройств поверх некогерентной оптики (incoherent.py).

Оптическое ядро IncoherentOpticalMul считает идеальные устройства: мощность светодиода
равна входу, пропускание маски равно W, детектор читает мощность без шума. Здесь
добавлено то, что меняется при переходе к стенду (results/devices, 28.10.2026):

  источник и маска — модуляторы с конечным числом уровней, гаммой и контрастом:
      уровень c = round(LUT⁻¹(v)·(L − 1)), пропускание T = T_min + (1 − T_min)·LUT(c/(L − 1)),
      LUT(u) = u^γ, T_min = 1/контраст. Цифра заранее обращает LUT (как при калибровке
      стенда), поэтому гамма проявляется только как неравномерный шаг квантования.
      Разброс элементов (подсветка, пиксели) — фиксированный множитель (1 + σ·ξ);
  детектор — дробовой шум, шум считывания, фон (рассеянный свет), насыщение и АЦП.
      Экспозиция выставлена так, что плоское поле (все входы 1, маска открыта)
      занимает долю full_scale от полной ёмкости.
  утечка закрытой маски T_min вычитается кадром с закрытой маской (как на стенде);
  кадр считается без шума, то есть усреднённым по многим повторам.

Округления пропускают градиент без изменений (straight-through), шум аддитивный,
поэтому модель годится и для обучения с учётом железа.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch as _torch
import torch.nn as _nn

from .incoherent import IncoherentOpticalMul as _Mul


def _ste_round(v: _torch.Tensor) -> _torch.Tensor:
    return v + (v.round() - v).detach()


@dataclass
class ModulatorConfig:
    """
    Модулятор интенсивности (ЖК, светодиод с ШИМ).
        levels:   число уровней (256 для 8 бит); 0 — без квантования.
        gamma:    показатель LUT(u) = u^γ (2,2 у типичного ЖК, 1 у светодиода с ШИМ).
        contrast: отношение открытого пропускания к закрытому; 0 — идеальный (T_min = 0).
        nonuni:   σ относительного разброса элементов (фиксированный рисунок).
    """
    levels: int = 0
    gamma: float = 1.0
    contrast: float = 0.0
    nonuni: float = 0.0

    @property
    def t_min(self) -> float:
        return 1.0 / self.contrast if self.contrast > 0 else 0.0

    @property
    def ideal(self) -> bool:
        return not self.levels and not self.t_min and not self.nonuni


@dataclass
class DetectorConfig:
    """
    Детектор (группа пикселей камеры или фотодиод на выход).
        full_well:  полная ёмкость детектора, e⁻; 0 — без шумов и насыщения.
        full_scale: доля ёмкости, которую занимает плоское поле (выбор экспозиции).
        read_noise: шум считывания, e⁻ (на детектор, уже с учётом биннинга).
        adc_bits:   разрядность АЦП на полную ёмкость; 0 — без квантования.
        background: рассеянный свет как доля плоского поля; вычитается по среднему,
                    остаётся его дробовой шум.
        frames:     число усредняемых кадров (шум делится на √frames).
    """
    full_well: float = 0.0
    full_scale: float = 0.8
    read_noise: float = 0.0
    adc_bits: int = 0
    background: float = 0.0
    frames: int = 1

    @property
    def ideal(self) -> bool:
        return not self.full_well


@dataclass
class DeviceConfig:
    source: ModulatorConfig = field(default_factory=ModulatorConfig)
    mask: ModulatorConfig = field(default_factory=ModulatorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    subtract_leak: bool = True
    seed: int = 0


class _Modulator(_nn.Module):
    def __init__(self, cfg: ModulatorConfig, shape, gen):
        super().__init__()
        self.cfg = cfg
        pattern = 1 + cfg.nonuni * _torch.randn(shape, generator=gen) if cfg.nonuni else _torch.ones(shape)
        self.register_buffer('pattern', pattern.clamp_min(0), persistent=False)

    def forward(self, v: _torch.Tensor, pattern: Optional[_torch.Tensor] = None) -> _torch.Tensor:
        c = self.cfg
        v = v.float().clamp(0., 1.)
        if c.levels:
            u = v.clamp_min(1e-12) ** (1 / c.gamma) if c.gamma != 1 else v
            u = _ste_round(u * (c.levels - 1)) / (c.levels - 1)
            v = u ** c.gamma if c.gamma != 1 else u
        t = c.t_min + (1 - c.t_min) * v
        return t if pattern is None else t * pattern


class IncoherentDevice(_nn.Module):
    """
    IncoherentOpticalMul с реальными устройствами. Контракт тот же:
        forward(input, other, response=None), response(other).
    """

    def __init__(self, mul: _Mul, cfg: DeviceConfig):
        super().__init__()
        self.mul, self.cfg = mul, cfg
        c = mul.config
        g = _torch.Generator().manual_seed(cfg.seed)
        self.src = _Modulator(cfg.source, (c.n_in,), g)
        self.mask = _Modulator(cfg.mask, (c.n_in, c.n_out), g)
        with _torch.no_grad():
            flat = mul(_torch.ones(1, 1, 1, c.n_in, device=mul._kc.device),
                       _torch.ones(1, 1, c.n_in, c.n_out, device=mul._kc.device))
        # Сигнал плоского поля: по нему выставлена экспозиция (максимум по детекторам).
        self.register_buffer('_flat', flat.max().detach().reshape(()), persistent=False)

    # Фиксированные рисунки разброса индексируются по центрированному окну, как _crop.
    def _patterns(self, w: int, k: int):
        c = self.mul.config
        pw, pk = (c.n_in - w) // 2, (c.n_out - k) // 2
        return self.src.pattern[pw:pw + w], self.mask.pattern[pw:pw + w, pk:pk + k]

    def response(self, other: _torch.Tensor):
        """
        Отклик маски для other: (R открытой маски, R закрытой маски или None).
        Второй нужен, чтобы вычесть утечку закрытых ячеек; его можно кэшировать вместе с первым.
        """
        _, pm = self._patterns(other.shape[-2], other.shape[-1])
        r = self.mul.response(self.mask(other, pm))
        leak = None
        if self.cfg.subtract_leak and self.cfg.mask.t_min:
            closed = (self.cfg.mask.t_min * pm).expand_as(other).contiguous()
            leak = self.mul.response(closed)
        return r, leak

    def detect(self, y: _torch.Tensor) -> _torch.Tensor:
        """Полный сигнал детектора (с утечкой маски) -> отсчёт в единицах выхода оптики."""
        d = self.cfg.detector
        if d.ideal:
            return y
        e = d.full_well * d.full_scale / self._flat           # e⁻ на единицу выхода оптики
        sig = y * e
        bg = d.background * d.full_well * d.full_scale
        var = sig.detach().clamp_min(0) + bg + d.read_noise ** 2
        sig = sig + _torch.randn_like(sig) * (var / d.frames).sqrt()
        sig = sig.clamp(max=d.full_well - bg)                  # насыщение: фон тоже занимает ёмкость
        if d.adc_bits:
            lsb = d.full_well / (2 ** d.adc_bits - 1)
            sig = _ste_round(sig / lsb) * lsb
        return sig / e

    def forward(self, input, other, response=None):
        ps, _ = self._patterns(other.shape[-2], other.shape[-1])
        x = self.src(input, ps)
        r, leak = self.response(other) if response is None else response
        y = self.detect(self.mul(x, other, response=r))
        if leak is not None:
            # Кадр с закрытой маской при том же входе, усреднённый (без шума).
            y = y - self.mul(x, other, response=leak)
        return y
