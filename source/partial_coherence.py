"""
partial_coherence.py — ветка partial-coherence: некогерентная и частично-когерентная
схема поверх существующего когерентного пропагаторного движка (репо
Optical_matrix_multiplication, source/).

Ключевая физика:
  * Каждый излучатель (VCSEL_i) сам с собой когерентен -> его поле на детекторе
    считается СУЩЕСТВУЮЩИМИ пропагаторами (комплексная амплитуда, точно).
  * Некогерентность/частичная когерентность входит ТОЛЬКО в момент суммирования
    на детекторе:
        I(y,t) = | sum_i E_i(y) * exp(i*phi_i(t)) |^2
               = sum_i |E_i|^2  +  sum_{i!=j} Re[ E_i E_j^* e^{i(phi_i-phi_j)} ]
    После усреднения детектором фактор при перекрёстном члене ij — это
    взаимная когерентность gamma_ij (0..1).

Режимы:
  * coherent   : gamma = 1  (один лазер + DMD; текущий сим как он есть)
  * incoherent : gamma = 0  (идеальная VCSEL-линейка; I = |T|^2 @ I_in)
  * partial    : произвольная матрица gamma (эрмитова, диагональ = 1)
  * beats      : явная временная симуляция биений для заданных расстроек
                 dnu_ij и окна интегрирования детектора (отвечает на вопрос
                 оптиков «какая нужна задержка считывания»).

Всё построено на передаточной матрице T[out_pixel, emitter]: она извлекается
ОДНИМ батчированным forward-проходом (identity по оси H) через немодифициро-
ванный OpticalMul, поэтому вся физика распространения — из основного репо.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from .config import Config
from .optical_mul import OpticalMul



# ============================================================================
# 1. Извлечение передаточной матрицы T
# ============================================================================

class TransferMatrix:
    """
    T[out, in]: комплексный отклик выходной плоскости (до детектора) на
    единичный вход в элементе in. Включает в себя всю цепочку prepare_vector ->
    propagator_one -> (* W) -> propagator_two, т.е. flip'ы и kron-разбиения
    учтены автоматически (пайплайн линеен по входному полю).

    Зависит от весовой матрицы W (правый операнд), поэтому кэшируется на тайл
    и пересчитывается при смене весов — в обучении амортизируется по батчу и
    по шагам между обновлениями.
    """

    def __init__(self, mul: OpticalMul, weights: torch.Tensor,
                 chunk: int = 256):
        """
        Args:
            mul:     немодифицированный OpticalMul из репо.
            weights: правая матрица (W, K) | (B, C, W, K) — элемент SLM.
            chunk:   размер чанка базисных векторов (память ~ chunk * n_out).
        """
        self.mul = mul
        if weights.dim() == 2:
            weights = weights[None, None]
        self.weights = weights
        self.vector_size = weights.size(-2)
        self._T = self._extract(chunk)

    @torch.no_grad()
    def _extract(self, chunk: int) -> torch.Tensor:
        W = self.vector_size
        mat_field = self.mul.prepare_matrix(self.weights)
        cols = []
        for s in range(0, W, chunk):
            e = min(s + chunk, W)
            basis = torch.eye(W)[s:e].view(1, 1, e - s, W)      # H = чанк базиса
            vec = self.mul.prepare_vector(basis)
            f = self.mul._propagator_one(vec, mat_field.shape[-2:])
            f = self.mul._propagator_two(f * mat_field,
                                         (mat_field.size(-2), 1))
            cols.append(f.squeeze(0).squeeze(0).squeeze(-1))     # (chunk, n_out)
        return torch.cat(cols, dim=0).transpose(0, 1).contiguous()  # (n_out, W)

    @property
    def T(self) -> torch.Tensor:
        """(n_out, vector_size), complex."""
        return self._T

    @property
    def intensity_kernel(self) -> torch.Tensor:
        """|T|^2 — вещественное ядро некогерентного режима."""
        return self._T.abs().square()

    # --- детектор: пуллинг и конвенции чтения --------------------------------
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """AvgPool по result_split, как в prepare_out, плюс финальный flip."""
        shape = x.shape
        y = self.mul._avg_pool(x.reshape(-1, 1, shape[-1]))
        return y.reshape(*shape[:-1], -1).flip(-1)

    def read_amplitude_legacy(self, field: torch.Tensor) -> torch.Tensor:
        """Конвенция репо: |E| -> avg_pool (амплитудное чтение)."""
        return self._pool(field.abs())

    def read_intensity(self, intensity: torch.Tensor,
                       sqrt_out: bool = False) -> torch.Tensor:
        """Физическая конвенция: детектор суммирует ИНТЕНСИВНОСТЬ по пикселю."""
        y = self._pool(intensity)
        return y.sqrt() if sqrt_out else y


# ============================================================================
# 2. Спецификация VCSEL-линейки
# ============================================================================

@dataclass
class VCSELArraySpec:
    """
    Всё, что определяет статистику взаимной когерентности линейки и полосу
    приёмного тракта. Это же — будущая спецификация закупки.
    """
    n_emitters: int
    wavelength: float = 850e-9          # м
    dnu_min: float = 2.5e9              # Гц, мин. попарная расстройка (by design)
    dnu_sigma: float = 5e9              # Гц, разброс расстроек (изготовление)
    linewidth: float = 100e6            # Гц, ширина линии одиночного VCSEL
    f_mod: float = 1e9                  # Гц, частота модуляции входа
    f_adc: float = 500e6                # Гц, частота выборки АЦП
    detector_bw: float = 500e6          # Гц, аналоговая полоса PD-тракта
    seed: int = 0

    @property
    def t_int(self) -> float:
        """Окно интегрирования одного отсчёта (прямоугольное), с."""
        return 1.0 / self.f_adc

    def sample_frequencies(self) -> torch.Tensor:
        """
        Оптические частоты излучателей (относительно nu_0): равномерная
        гребёнка с шагом >= dnu_min + гауссов джиттер. Модель «спейсинг by
        design + технологический разброс».
        """
        g = torch.Generator().manual_seed(self.seed)
        base = torch.arange(self.n_emitters, dtype=torch.float64) * self.dnu_min
        jitter = torch.randn(self.n_emitters, generator=g,
                             dtype=torch.float64) * self.dnu_sigma
        return base + jitter

    def gamma_from_spectra(self) -> torch.Tensor:
        """
        Матрица |gamma_ij| после прямоугольного окна интегрирования T_int и
        однополюсного фильтра детектора B:
            gamma_ij = |sinc(dnu_ij * T_int)| / sqrt(1 + (dnu_ij / B)^2)
        Дополнительно каждый член умножается на exp(-pi*linewidth*T_int) —
        фазовая диффузия двух независимых лазеров за окно (консервативная
        оценка сверху опущена: берём средний фактор).
        """
        nu = self.sample_frequencies()
        dnu = (nu[:, None] - nu[None, :]).abs()
        sinc = torch.sinc(dnu * self.t_int)                # torch.sinc = sin(pi x)/(pi x)
        lpf = 1.0 / torch.sqrt(1.0 + (dnu / self.detector_bw) ** 2)
        gamma = (sinc.abs() * lpf).to(torch.float32)
        gamma.fill_diagonal_(1.0)
        return gamma

    def required_dnu_min(self, margin: float = 1.0) -> float:
        """
        Правило спецификации: биение ij не должно попадать в полосу тракта
        с учётом модуляционных боковых полос:
            dnu_min >= f_adc + 2*f_mod  (плюс запас margin, Гц-множитель)
        """
        return margin * (self.f_adc + 2.0 * self.f_mod)


# ============================================================================
# 3. Частично-когерентное умножение
# ============================================================================

class PartialCoherentMul:
    """
    Обёртка над TransferMatrix с тремя стационарными режимами и явной
    временной симуляцией биений.

    Соглашение о входе:
      * coherent            : x — амплитуды (как в текущем симе);
      * incoherent / partial: x — ИНТЕНСИВНОСТИ излучателей, x >= 0
                              (VCSEL кодирует значение мощностью).
    """

    def __init__(self, tm: TransferMatrix):
        self.tm = tm

    # --- стационарные режимы -------------------------------------------------
    def coherent(self, x: torch.Tensor, legacy_read: bool = True) -> torch.Tensor:
        """|T @ x|, конвенция чтения репо. Кросс-чек против mul(A, W)."""
        f = torch.einsum('oi,...i->...o', self.tm.T, x.cfloat())
        if legacy_read:
            return self.tm.read_amplitude_legacy(f)
        return self.tm.read_intensity(f.abs().square())

    def incoherent(self, x_int: torch.Tensor,
                   sqrt_out: bool = False) -> torch.Tensor:
        """Идеальная линейка: I_out = |T|^2 @ I_in. Обычный вещественный матмул."""
        i = torch.einsum('oi,...i->...o', self.tm.intensity_kernel,
                         x_int.float())
        return self.tm.read_intensity(i, sqrt_out=sqrt_out)

    def partial(self, x_int: torch.Tensor, gamma: torch.Tensor,
                phases: Optional[torch.Tensor] = None,
                sqrt_out: bool = False) -> torch.Tensor:
        """
        I(y) = Re[ e_y^T (Gamma .* P) e_y^* ],  e_y,i = T[y,i] * sqrt(I_i),
        P_ij = exp(i(phi_i - phi_j)) — мгновенные фазы (по умолчанию нулевые:
        худший случай синфазного сложения; для ансамблевого среднего фазы
        сэмплируются снаружи).
        gamma: (N, N), эрмитова по модулю, diag = 1. gamma=1 -> coherent
        (в интенсивностном чтении), gamma=0 -> incoherent.
        """
        a = x_int.float().clamp_min(0).sqrt()
        E = self.tm.T[None, ...] * a[..., None, :]           # (..., n_out, N)
        G = gamma.cfloat()
        if phases is not None:
            ph = torch.exp(1j * phases)
            G = G * ph[:, None] * ph[None, :].conj()
        i = torch.einsum('...oi,ij,...oj->...o', E, G, E.conj()).real
        return self.tm.read_intensity(i.clamp_min(0), sqrt_out=sqrt_out)

    # --- явная временная симуляция биений -----------------------------------
    def beats(self, x_int: torch.Tensor, spec: VCSELArraySpec,
              n_windows: int = 1, oversample: int = 64,
              seed: int = 0) -> torch.Tensor:
        """
        Честный прогон по времени: I(t) с фазами 2*pi*nu_i*t + phi0_i +
        фазовая диффузия (винеровский процесс с D = pi*linewidth), усреднение
        прямоугольным окном T_int; n_windows окон усредняются дополнительно
        (это и есть «задержка считывания» = n_windows / f_adc).
        """
        g = torch.Generator().manual_seed(seed)
        nu = spec.sample_frequencies()                        # (N,) float64
        N = spec.n_emitters
        a = x_int.float().clamp_min(0).sqrt()
        E = self.tm.T * a[None, :]                            # (n_out, N)

        dt = spec.t_int / oversample
        t = torch.arange(oversample, dtype=torch.float64) * dt
        acc = torch.zeros(self.tm.T.size(0))
        phi0 = torch.rand(N, generator=g, dtype=torch.float64) * 2 * math.pi
        # фазовая диффузия: приращения на каждый под-шаг
        diff_std = math.sqrt(2 * math.pi * spec.linewidth * dt)
        for w in range(n_windows):
            t_w = t + w * spec.t_int
            phase = 2 * math.pi * nu[None, :] * t_w[:, None] + phi0[None, :]
            walk = (torch.randn(oversample, N, generator=g,
                                dtype=torch.float64) * diff_std).cumsum(0)
            ph = torch.exp(1j * (phase + walk)).to(torch.complex64)  # (S, N)
            f = torch.einsum('oi,si->so', E, ph)              # (S, n_out)
            acc += f.abs().square().mean(0)
            phi0 = (phase[-1] + walk[-1])                     # непрерывность фаз
        return self.tm.read_intensity(acc / n_windows)


EPS_ENCODE = 1e-12   # защита градиента d(sqrt)/dx = 1/(2 sqrt x) в нуле


def encode_amplitude(values: torch.Tensor) -> torch.Tensor:
    """
    Значение -> амплитудное пропускание / амплитуда поля.
    Реализация фикса оптиков: в оптику подаётся sqrt(значения), поэтому
    детектор, меряющий квадрат, сразу читает значение без пост-обработки.
    Требует неотрицательности (схема сдвигов её уже обеспечивает).
    """
    if (values < 0).any():
        raise ValueError("sqrt-кодировка требует values >= 0 "
                         "(используйте схему сдвигов в неотрицательный домен)")
    return values.clamp_min(EPS_ENCODE).sqrt()


def encode_intensity(values: torch.Tensor) -> torch.Tensor:
    """
    Значение -> интенсивность излучателя. Тождество: VCSEL кодирует значение
    мощностью напрямую. Существует ради симметрии с encode_amplitude и
    явности намерения в коде вызывающей стороны.
    """
    if (values < 0).any():
        raise ValueError("интенсивностная кодировка требует values >= 0")
    return values


def decode_intensity(detected: torch.Tensor) -> torch.Tensor:
    """
    Показания детектора -> значения. Тождество: в некогерентной схеме с
    sqrt-кодировкой пост-обработка корнем НЕ нужна (в отличие от когерентной
    схемы, где стоял |E| -> ... -> sqrt). Функция существует, чтобы это
    свойство было видно в коде, а не только в комментарии.
    """
    return detected


class IncoherentMVM:
    """
    Готовый к употреблению некогерентный MVM с зашитой кодировкой.

    Контракт: и веса, и вход, и выход — в домене ЗНАЧЕНИЙ. Кодировка
    (sqrt на веса, интенсивность на вход, отсутствие корня на выходе)
    выполняется внутри, ошибиться нельзя.

        mvm = IncoherentMVM(mul, W)      # W — значения, не sqrt(W)
        y   = mvm(x)                     # y ~ x @ W с точностью до калибр. скаляра

    Скаляр c — физический коэффициент передачи тракта (потери, апертура),
    в железе снимается калибровкой по реперным матрицам; здесь считается
    один раз по единичному отклику и хранится в .gain.
    """

    def __init__(self, mul: OpticalMul, weights: torch.Tensor, chunk: int = 256,
                 calibrate: bool = True):
        self.tm = TransferMatrix(mul, encode_amplitude(weights), chunk=chunk)
        self.pcm = PartialCoherentMul(self.tm)
        self.weights = weights
        self.gain = 1.0
        if calibrate:
            self.gain = self._calibrate()

    @torch.no_grad()
    def _calibrate(self) -> float:
        """Коэффициент передачи по реперному входу (единичный вектор значений)."""
        x = torch.ones(self.tm.vector_size)
        y = self.pcm.incoherent(encode_intensity(x)).flatten()
        y_ref = x @ self.weights
        return ((y @ y_ref) / (y_ref @ y_ref)).item()

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        y = self.pcm.incoherent(encode_intensity(values))
        return decode_intensity(y) / self.gain


# ============================================================================
# 4. Утилиты анализа: карта ошибки и закон 1/sqrt(M)
# ============================================================================

def crosstalk_error(y_test: torch.Tensor, y_ref: torch.Tensor) -> float:
    """Относительная ошибка ||y - y_ref|| / ||y_ref|| (метрика ветки)."""
    return (y_test - y_ref).norm().item() / y_ref.norm().item()


@torch.no_grad()
def epsilon_vs_averaging(pcm: PartialCoherentMul, x_int: torch.Tensor,
                         gamma_scalar: float,
                         m_list: Sequence[int], seed: int = 0,
                         n_repeats: int = 1) -> list[float]:
    """
    Ансамблевое усреднение M независимых реализаций фаз при фиксированной
    остаточной когерентности gamma_scalar. Возвращает eps(M) — проверка
    закона eps ~ gamma * C / sqrt(M) и калибровка константы C для формулы
    задержки t_delay = M * tau_decorr.
    """
    N = pcm.tm.T.size(1)
    gamma = torch.full((N, N), float(gamma_scalar))
    gamma.fill_diagonal_(1.0)
    y_ref = pcm.incoherent(x_int)
    g = torch.Generator().manual_seed(seed)
    out = []
    for m in m_list:
        errs = []
        for _ in range(n_repeats):
            acc = None
            for _ in range(m):
                ph = torch.rand(N, generator=g) * 2 * math.pi
                y = pcm.partial(x_int, gamma, phases=ph)
                acc = y if acc is None else acc + y
            errs.append(crosstalk_error(acc / m, y_ref))
        out.append(sum(errs) / len(errs))
    return out


@torch.no_grad()
def wavelength_sensitivity(config_factory: Callable[[float], Config],
                           weights: torch.Tensor,
                           wavelengths: Sequence[float]) -> list[float]:
    """
    Чувствительность |T|^2 к длине волны: строим T(lambda) через фабрику
    конфигов и меряем относительный сдвиг ядра от базовой lambda[0].
    Вход в допуск на спектральный разброс линейки (пм–нм масштаб).
    """
    base = TransferMatrix(OpticalMul(config_factory(wavelengths[0])), weights)
    k0 = base.intensity_kernel
    out = [0.0]
    for lam in wavelengths[1:]:
        tm = TransferMatrix(OpticalMul(config_factory(lam)), weights)
        out.append((tm.intensity_kernel - k0).norm().item() / k0.norm().item())
    return out