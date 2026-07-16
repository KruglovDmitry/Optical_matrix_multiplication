"""
optics.py — переиспользуемая обёртка над оптическим симулятором
для встраивания оптического матричного умножения в трансформер.

Ключевая идея — переключаемость: каждый матмул можно независимо гонять либо
через оптический sim, либо через обычный torch, а режим градиента выбирается
одним параметром движка.

Режимы градиента (backward=):
  * 'twin'     (по умолчанию) — PAT-S: forward через (шумный) физический тракт,
                straight-through только вокруг sim; градиент — через цифровой
                двойник (матмул × калибровка) и цифровой конвейер нормировок.
                Эмпирически эквивалентен полному autograd (зазор 0.2–3% ppl),
                ~2x быстрее, не хранит активации пропагатора. Контракт железа:
                устройству нужен только forward + свежая калибровка.
  * 'autograd' — полный autograd сквозь пропагаторы sim. Нужен для линии
                обучаемых оптических элементов (add-parameters) и как эталон.
  Полный оптический офлоад (градиенты через оптику + знаковый дизеринг)
  исследован и удалён из кода как неосновной: результаты и реализация — в
  git-тегах ob-dither / optical-backward и в отчёте (+11–15% ppl).

История: режимы PAT (чистый двойник) и PAT-N (двойник со статистическим шумом)
удалены как проигравшие абляции — см. git-тег pat-s-validated и write-up.
Механизм: адаптации весов к шуму нужен градиент, коррелированный с РЕАЛИЗАЦИЕЙ
шума forward-прохода; она входит через измеренную невязку и нормировки a·b.

Компоненты:
  * OpticalNoiseModel — модель неидеальностей устройства (SLM, детектор, АЦП,
                        белый шум усиления, медленный дрейф);
  * optics_matmul     — примитив (сдвиг → нормировка → sim → масштаб → поправки);
  * OpticalEngine     — движок: один omm.OpticalMul, калибровки (скаляр gain +
                        попиксельные flat-field карты), тайлинг с батчированием;
  * mm(...)           — диспетчер optical/torch;
  * OpticLinear, OpticalAttention — слои с переключаемым бэкендом.

Контракт симулятора (README): вход — два 4D-тензора
    left (B, C, H, W) и right (B, C, W, K)  →  выход (B, C, H, K).
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from einops import rearrange
from .config import Config
from .optical_mul import OpticalMul

# Физическая «усилка» симулятора: sim(X,Y) ~ 3.44e-3 * (X@Y). gain_inv её компенсирует.
# ВНИМАНИЕ: скаляр зависит от апертуры И от формы матриц (краевые эффекты:
# полная апертура 512 даёт ~2.46e-3, малые центральные матрицы ~3.43e-3).
# Точный масштаб на форму обеспечивают flat-field карты; скалярный gain задаёт
# лишь единицы полной шкалы детектора (full_scale = k/gain) для шумовой модели.
# Менять процедуру калибровки среди серии экспериментов нельзя — поплывут
# единицы сигм. calibrate_gain() сохранён в прежнем виде ради непрерывности.
SIM_GAIN_INV = 1.0 / 3.44e-3

# Именованные профили устройства — единый источник правды для геометрии.
# Числа в скриптах не дублируются: движок и эксперименты ссылаются на профиль,
# явные аргументы поверх профиля выигрывают.
DEVICE_PROFILES = {
    # исторический профиль программы d13 (воспроизводимость старых серий);
    # ВНИМАНИЕ: вне чистой зоны точность падает, см. geometry_probe
    'd13':      dict(distance=0.01, lens_size=8192, tile_size=None,
                     tile_batch=None),
    # быстрый чистый профиль кампании d12: N_F~2 на тайле, ошибка ~0.25%
    'fast_d12': dict(distance=0.05, lens_size=8192, tile_size=128,
                     tile_batch=16),
    # геометрия прототипа от оптиков, численно СОШЕДШАЯСЯ конфигурация:
    # окно 32768 обязательно (16384 недосэмплирует поле на z=0.15 — ошибки
    # x100; проверено geometry_probe 2026-07-16). Тайл 128 — равномерные
    # ~0.2-0.3% по всем формам; окно влияет только на предвычисление
    # операторов, рантайм и память вызовов от него не зависят.
    'proto':    dict(distance=0.15, lens_size=32768, tile_size=128,
                     tile_batch=16),
}


class OpticalNoiseModel(nn.Module):
    """
    Модель неидеальностей физического устройства. Вставляется в двух точках
    optics_matmul — ровно там, где живёт железо:

      modulate(x) — вход SLM/DOE (нормированные неотрицательные матрицы в [0,1]):
        * input_bits  — битность модулятора (квантование уровней, STE-градиент);
        * input_sigma — мультипликативный шум модуляции (ошибка уровня пикселя).

      detect(i, full_scale) — сырой выход детектора ДО восстановления масштаба
      (* a * b * gain), поэтому шум честно усиливается при вычитании поправок:
        * gain_sigma   — мультипликативная флуктуация усиления тракта;
        * shot_photons — дробовой шум: число фотонов на полной шкале
                         (std ∝ sqrt(I)); None = выключен;
        * read_sigma   — аддитивный шум чтения, в долях полной шкалы детектора
                         (1e-3 ≈ 60 дБ SNR относительно FS);
        * output_bits  — битность АЦП (квантование к полной шкале, STE-градиент).

    Полная шкала full_scale = k / gain_inv: максимум сырой интенсивности для
    нормированных входов при контракции длины k.

    Шум — свойство физики устройства, а не регуляризация, поэтому по умолчанию
    активен и в train, и в eval (perplexity меряется «как на железе»).
    Отключается флагом enabled или eval_noise=False (для чистых замеров).

    Квантование через straight-through estimator: forward — ступенька,
    backward — тождественный градиент, иначе обучение через оптику умрёт.
    """
    def __init__(self, input_bits=None, input_sigma=0.0,
                 gain_sigma=0.0, shot_photons=None,
                 read_sigma=0.0, output_bits=None,
                 gain_drift_sigma=0.0, gain_drift_tau=1000.0,
                 enabled=True, eval_noise=True):
        super().__init__()
        self.input_bits = input_bits
        self.input_sigma = float(input_sigma)
        self.gain_sigma = float(gain_sigma)
        self.shot_photons = shot_photons
        self.read_sigma = float(read_sigma)
        self.output_bits = output_bits
        # --- Медленный дрейф усиления (Орнштейна–Уленбека) ---
        # В отличие от белого gain_sigma, дрейф коррелирован во времени и НЕ
        # усредняется обучением по батчам — честный прокси температурного
        # дрейфа и плавания мощности лазера на реальном столе.
        #   gain_drift_sigma — стационарное std дрейфа (доли усиления);
        #   gain_drift_tau   — время корреляции В ВЫЗОВАХ sim (не итерациях!
        #     вызовов за итерацию ~ layers × sites × tiles; для конфигурации
        #     свипа (2 слоя, всё оптическое) это ~16+ вызовов на forward).
        # recalibrate() фиксирует текущее значение дрейфа как «известное»
        # драйверу: применяемая ошибка = (1 + drift) / (1 + drift_ref),
        # т.е. остаточный дрейф с момента последней калибровки.
        self.gain_drift_sigma = float(gain_drift_sigma)
        self.gain_drift_tau = float(gain_drift_tau)
        self._drift = 0.0
        self._drift_ref = 0.0
        self.enabled = enabled
        self.eval_noise = eval_noise

    def _step_drift(self):
        rho = math.exp(-1.0 / self.gain_drift_tau)
        self._drift = (rho * self._drift
                       + math.sqrt(max(1.0 - rho * rho, 0.0))
                       * self.gain_drift_sigma * torch.randn(()).item())

    def recalibrate(self):
        """Калибровка драйвера: текущий дрейф становится «известным» и компенсируется."""
        self._drift_ref = self._drift

    def _active(self):
        return self.enabled and (self.training or self.eval_noise)

    @staticmethod
    def _ste(x, x_q):
        # Прямой проход — квантованное значение, градиент — как у identity.
        return x + (x_q - x).detach()

    def modulate(self, x):
        if not self._active():
            return x
        if self.input_bits is not None:
            levels = (1 << self.input_bits) - 1
            x = self._ste(x, torch.round(x.clamp(0., 1.) * levels) / levels)
        if self.input_sigma > 0:
            x = (x * (1. + torch.randn_like(x) * self.input_sigma)).clamp(0., 1.)
        return x

    def detect(self, i_raw, full_scale):
        if not self._active():
            return i_raw
        y = i_raw
        if self.gain_drift_sigma > 0:
            self._step_drift()
            y = y * ((1.0 + self._drift) / (1.0 + self._drift_ref))
        if self.gain_sigma > 0:
            y = y * (1. + torch.randn_like(y) * self.gain_sigma)
        if self.shot_photons:
            # std фотонного шума в единицах шкалы: fs * sqrt((I/fs) / N_ph)
            std = full_scale * torch.sqrt(
                y.detach().clamp_min(0.) / full_scale / float(self.shot_photons))
            y = y + torch.randn_like(y) * std
        if self.read_sigma > 0:
            y = y + torch.randn_like(y) * (self.read_sigma * full_scale)
        if self.output_bits is not None:
            levels = (1 << self.output_bits) - 1
            u_q = torch.round((y / full_scale).clamp(0., 1.) * levels) / levels
            y = self._ste(y, u_q * full_scale)
        return y

    def extra_repr(self):
        return (f"input_bits={self.input_bits}, input_sigma={self.input_sigma}, "
                f"gain_sigma={self.gain_sigma}, shot_photons={self.shot_photons}, "
                f"read_sigma={self.read_sigma}, output_bits={self.output_bits}, "
                f"gain_drift_sigma={self.gain_drift_sigma}, "
                f"gain_drift_tau={self.gain_drift_tau}, "
                f"enabled={self.enabled}, eval_noise={self.eval_noise}")


def _twin_pipeline(A, B, delta, gain, ffmap, eps=1e-8):
    """Цифровой конвейер двойника с инъекцией сохранённой невязки δ.
    Вызывается дважды: в forward (значение) и в backward (граф) — дёшево,
    т.к. это чистая цифра без sim. δ несёт реализацию физики/шума."""
    k = A.shape[-1]
    sa = torch.clamp(-A.amin(dim=-1, keepdim=True), min=0)
    sb = torch.clamp(-B.amin(dim=-2, keepdim=True), min=0)
    P, Q = A + sa, B + sb
    a = P.amax(dim=-1, keepdim=True).clamp_min(eps)
    b = Q.amax(dim=-2, keepdim=True).clamp_min(eps)
    ref = torch.matmul(P / a, Q / b) / gain
    if ffmap is not None:
        ref = ref / ffmap.clamp_min(1e-12)
    raw = ref + delta
    if ffmap is not None:
        raw = raw * ffmap
    PQ = raw * a * b * gain
    corr_a = sb * A.sum(dim=-1, keepdim=True)
    corr_b = sa * B.sum(dim=-2, keepdim=True)
    return PQ - corr_a - corr_b - k * sa * sb


class _TwinMM(torch.autograd.Function):
    """
    Memory-light twin (backward='twin'): вместо хранения ~6 промежуточных
    тензоров конвейера на каждый тайл (OOM на глубоких моделях) сохраняем
    только входы (view активаций) и невязку δ = raw_phys − ref. Backward
    пересчитывает цифровой конвейер под enable_grad — дёшево относительно
    sim; sim НЕ перезапускается; градиент идентичен прежней sim_st-ветке
    (тот же граф; значение raw = ref + δ = физическое, реализационный канал
    через a·b сохранён).
    """
    @staticmethod
    def forward(ctx, A, B, engine):
        gain, noise, sim = engine.gain, engine.noise, engine.sim
        ffmap = None
        if engine.flat_field:
            ffmap = engine._get_ff_map(A.shape[-2], A.shape[-1], B.shape[-1],
                                       A.device, torch.float32)
        k = A.shape[-1]
        with torch.no_grad():
            sa = torch.clamp(-A.amin(dim=-1, keepdim=True), min=0)
            sb = torch.clamp(-B.amin(dim=-2, keepdim=True), min=0)
            P, Q = A + sa, B + sb
            a = P.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            b = Q.amax(dim=-2, keepdim=True).clamp_min(1e-8)
            Pn, Qn = P / a, Q / b
            Pm = noise.modulate(Pn) if noise is not None else Pn
            Qm = noise.modulate(Qn) if noise is not None else Qn
            raw_phys = sim(Pm, Qm)
            if noise is not None:
                raw_phys = noise.detect(raw_phys, full_scale=k / gain)
            ref = torch.matmul(Pn, Qn) / gain
            if ffmap is not None:
                ref = ref / ffmap.clamp_min(1e-12)
            delta = raw_phys - ref
            out = _twin_pipeline(A, B, delta, gain, ffmap)
            if engine.collect_stats:
                s = engine.stats
                ff = ffmap if ffmap is not None else 1.0
                amp = ((raw_phys * ff * a * b * gain).norm()
                       / out.norm().clamp_min(1e-12)).item()
                s['amp_sum'] = s.get('amp_sum', 0.0) + amp
                s['amp_max'] = max(s.get('amp_max', 0.0), amp)
                s['calls'] = s.get('calls', 0) + 1
        ctx.save_for_backward(A, B, delta)
        ctx.gain, ctx.ffmap = gain, ffmap
        return out

    @staticmethod
    def backward(ctx, grad_out):
        A, B, delta = ctx.saved_tensors
        with torch.enable_grad():
            A_ = A.detach().requires_grad_(A.requires_grad)
            B_ = B.detach().requires_grad_(B.requires_grad)
            out = _twin_pipeline(A_, B_, delta, ctx.gain, ctx.ffmap)
            wanted = [t for t in (A_, B_) if t.requires_grad]
            gs = list(torch.autograd.grad(out, wanted, grad_out))
        gA = gs.pop(0) if A.requires_grad else None
        gB = gs.pop(0) if B.requires_grad else None
        return gA, gB, None


def optics_matmul(sim, A, B, eps=1e-8, gain=1.0, noise=None, stats=None,
                  ffmap=None):
    """
    Приближённо вычисляет A @ B через неотрицательный оптический sim
    (полный autograd-путь; используется в режиме backward='autograd' и на
    инференсе). Обучение в режиме 'twin' идёт через _TwinMM.

    Оптика принимает только неотрицательные интенсивности, поэтому:
      1) строки A и столбцы B сдвигаются в неотрицательную область (sa, sb);
      2) сдвинутые P, Q нормируются в [0,1];
      3) sim перемножает нормированные матрицы, масштаб восстанавливается;
      4) три поправки за сдвиг вычитаются аналитически.

    noise — OpticalNoiseModel или None; ffmap — flat-field карта (цифровой
    постпроцессинг после детектора); stats — аудит усиления шума.
    """
    k = A.shape[-1]
    sa = torch.clamp(-A.amin(dim=-1, keepdim=True), min=0)
    sb = torch.clamp(-B.amin(dim=-2, keepdim=True), min=0)
    P, Q = A + sa, B + sb
    a = P.amax(dim=-1, keepdim=True).clamp_min(eps)
    b = Q.amax(dim=-2, keepdim=True).clamp_min(eps)
    Pn, Qn = P / a, Q / b
    if noise is not None:
        Pn, Qn = noise.modulate(Pn), noise.modulate(Qn)
    raw = sim(Pn, Qn)                       # сырые интенсивности детектора
    if noise is not None:
        raw = noise.detect(raw, full_scale=k / gain)
    if ffmap is not None:
        raw = raw * ffmap
    PQ = raw * a * b * gain
    corr_a = sb * A.sum(dim=-1, keepdim=True)
    corr_b = sa * B.sum(dim=-2, keepdim=True)
    out = PQ - corr_a - corr_b - k * sa * sb
    if stats is not None:
        with torch.no_grad():
            amp = (PQ.norm() / out.norm().clamp_min(1e-12)).item()
            stats['amp_sum'] = stats.get('amp_sum', 0.0) + amp
            stats['amp_max'] = max(stats.get('amp_max', 0.0), amp)
            stats['calls'] = stats.get('calls', 0) + 1
    return out


class OpticalEngine(nn.Module):
    """
    Единый источник правды для оптических матмулов: один omm.OpticalMul, конфиг
    апертуры и калибровки. Создаётся один раз на модель и передаётся во все слои.

    matmul(a, b) считает a @ b по последним двум осям:
      * a:[...,M,K], b:[...,K,N] (или b — 2D-вес [K,N], broadcast по батчу);
      * все размеры <= size → один optics_matmul;
      * иначе тайлинг: по M/N — блоки, по контракции K — чанки с суммированием.
        Тайлы (i,j) независимы и БАТЧИРУЮТСЯ в ведущую ось sim группами по
        tile_batch (края дозаполняются нулями до единого размера — поправки
        от нулей точны, а единая форма означает одну flat-field карту на все
        тайлы). tile_batch=None (по умолчанию) — последовательный путь.
        Батчирование выгодно на GPU (меньше запусков ядер, выше утилизация);
        на CPU оно МЕДЛЕННЕЕ из-за паддинга. Перед включением провалидируй
        на своей карте (см. бенчмарк в experiments).
        Тайлинг физически честен: реальная апертура фиксирована.

    backward: 'twin' (PAT-S, по умолчанию) | 'autograd' | 'optical'.
    """
    def __init__(self, size=512, pixel_size=3.6e-6, gain_inv=SIM_GAIN_INV,
                 splits=2, distance=0.01, noise=None, flat_field=True,
                 ff_trials=8, backward='twin', tile_batch=None,
                 tile_size=None, lens_size=8192, rows_per_call=4096):
        super().__init__()
        assert backward in ('twin', 'autograd'), backward
        self.size = size
        # РАБОЧАЯ АПЕРТУРА: точность симулятора рушится к краям поля
        # (драйверы — поперечные ширины K и N: рел. ошибка на центрированных
        # данных ~1% при 64, ~10% при 128, ~100% при 256, >2000% при 512).
        # tile_size принудительно нарезает все матмулы на тайлы ≤ tile_size
        # в чистой центральной зоне. Для сравнимых серий (лестница масштабов)
        # tile_size обязан быть одинаковым во всех ранах — иначе «размер
        # модели» спутается с «качеством устройства». None = вся апертура
        # (поведение d13; формы d13 несли до ~40% детерминированной ошибки,
        # которую сеть абсорбировала адаптацией — задокументированный факт).
        self.tile_size = min(tile_size or size, size)
        # Бюджет СТРОК на один sim-вызов: каждая строка левой матрицы — это
        # отдельное поле, и транзиентная память пропагатора ~ строки × ширина².
        # Пары тайлов (tile_batch) не учитывают ведущую ось (у внимания это
        # batch×heads, в разы больше, чем у FF) — группировка режется так,
        # чтобы p · C · tile_size <= rows_per_call. 4096 строк ~ 6-12 ГБ
        # транзиента на апертуре 512; поднимайте при свободной памяти.
        self.rows_per_call = rows_per_call
        self.gain = gain_inv
        # Шумовая модель устройства (OpticalNoiseModel или None — идеальный sim).
        self.noise = noise
        # Аудит усиления шума: включай точечно, читай через pop_stats().
        self.collect_stats = False
        self.stats = {}
        # Flat-field: фиксированная позиционная неоднородность усиления тракта
        # (~0.3–1%) после вычитания поправок раздувается сокращением до 10%+
        # на центрированных данных. Карты калибруются лениво на каждую форму
        # (M,K,N) усреднением ff_trials прогонов (аналог усреднения кадров) и
        # применяются в цифровом постпроцессинге. КРИТИЧНО: карта участвует и
        # в двойнике backward='twin' — двойник обязан знать устройство
        # настолько, насколько его знает калибровка.
        self.flat_field = flat_field
        self.ff_trials = ff_trials
        self._ff_maps = {}
        self.backward = backward
        self.tile_batch = tile_batch
        self.sim = OpticalMul(
            Config(right_matrix_count_columns=size,
                   right_matrix_count_rows=size,
                   right_matrix_width=pixel_size * size,
                   right_matrix_height=pixel_size * size,
                   min_height_gap=pixel_size,
                   right_matrix_split_x=splits,
                   right_matrix_split_y=splits,
                   left_matrix_split_x=splits,
                   left_matrix_split_y=splits,
                   result_matrix_split=splits,
                   distance=distance,
                   lens_size=lens_size)
        )

    @staticmethod
    def _to_4d(t):
        # Симулятору нужен ровно 4D. Активация [B,T,K]->(1,B,T,K); 2D-вес [K,N]->(1,1,K,N).
        if t.dim() == 2:
            return t[None, None]
        if t.dim() == 3:
            return t[None]
        return t

    @torch.no_grad()
    def _get_ff_map(self, M, K, N, device, dtype):
        """Ленивая flat-field карта для формы (M,K,N): mean(ref / (sim*gain))."""
        key = (M, K, N)
        if key not in self._ff_maps:
            was = self.noise
            self.noise = None      # на sim калибруемся чистым трактом;
            acc = None             # на устройстве — усреднением кадров
            for _ in range(self.ff_trials):
                P = torch.rand(1, 1, M, K, device=device, dtype=dtype)
                Q = torch.rand(1, 1, K, N, device=device, dtype=dtype)
                m = torch.matmul(P, Q) / (self.sim(P, Q) * self.gain).clamp_min(1e-20)
                acc = m if acc is None else acc + m
            self.noise = was
            self._ff_maps[key] = (acc / self.ff_trials)
        m = self._ff_maps[key]
        if m.device != device:
            m = m.to(device); self._ff_maps[key] = m
        return m

    def _mm(self, a4, b4):
        ffmap = (self._get_ff_map(a4.shape[-2], a4.shape[-1], b4.shape[-1],
                                  a4.device, torch.float32)
                 if self.flat_field else None)
        stats = self.stats if self.collect_stats else None
        # twin: memory-light двойник (_TwinMM), только когда строится граф;
        # инференс и режим autograd — обычный путь без цифрового матмула
        # (в autograd градиент идёт сквозь sim).
        if (self.backward == 'twin' and torch.is_grad_enabled()
                and (a4.requires_grad or b4.requires_grad)):
            return _TwinMM.apply(a4, b4, self)
        ffmap = (self._get_ff_map(a4.shape[-2], a4.shape[-1], b4.shape[-1],
                                  a4.device, torch.float32)
                 if self.flat_field else None)
        return optics_matmul(self.sim, a4, b4, gain=self.gain,
                             noise=self.noise, stats=stats, ffmap=ffmap)

    def _tiled_sequential(self, a4, b4):
        s = self.tile_size
        M, K = a4.shape[-2], a4.shape[-1]
        N = b4.shape[-1]
        rows = []
        for i in range(0, M, s):
            cols = []
            for j in range(0, N, s):
                acc = None
                for kk in range(0, K, s):
                    p = self._mm(a4[..., i:i + s, kk:kk + s],
                                 b4[..., kk:kk + s, j:j + s])
                    acc = p if acc is None else acc + p
                cols.append(acc)
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)

    def _tiled_matmul(self, a4, b4):
        s = self.tile_size
        M, K = a4.shape[-2], a4.shape[-1]
        N = b4.shape[-1]
        # Батчированный путь требует свободной ведущей оси (наш стандартный
        # лэйаут: активации [1,B,T,K], веса [1,1,K,N]); иначе — последовательно.
        if (self.tile_batch is None
                or a4.shape[0] != 1 or b4.shape[0] != 1):
            return self._tiled_sequential(a4, b4)
        n_i, n_j, n_k = -(-M // s), -(-N // s), -(-K // s)
        if n_i * n_j == 1:
            return self._tiled_sequential(a4, b4)
        # Нулевое дозаполнение до кратности s: нулевые строки/столбцы дают
        # нулевой вклад в контракцию, поправки на дозаполненных матрицах точны.
        Ap = F.pad(a4, (0, n_k * s - K, 0, n_i * s - M))
        Bp = F.pad(b4, (0, n_j * s - N, 0, n_k * s - K))
        Ca, Cb = Ap.shape[1], Bp.shape[1]
        # Тайлы: A -> [n_i, n_k, Ca, s, s], B -> [n_k, n_j, Cb, s, s]
        At = Ap[0].reshape(Ca, n_i, s, n_k, s).permute(1, 3, 0, 2, 4)
        Bt = Bp[0].reshape(Cb, n_k, s, n_j, s).permute(1, 3, 0, 2, 4)
        P = n_i * n_j
        ii = torch.arange(P, device=a4.device) // n_j
        jj = torch.arange(P, device=a4.device) % n_j
        # шаг группы: не больше tile_batch пар И не больше rows_per_call строк
        Ca = At.shape[2]
        p_step = max(1, min(self.tile_batch,
                            self.rows_per_call // max(1, Ca * s)))
        out = None
        for kk in range(n_k):
            acc_k = []
            for p0 in range(0, P, p_step):
                sl = slice(p0, min(p0 + p_step, P))
                left = At[ii[sl], kk]          # [p, Ca, s, s]
                right = Bt[kk, jj[sl]]         # [p, Cb, s, s]
                acc_k.append(self._mm(left, right))
            part = torch.cat(acc_k, dim=0)     # [P, C, s, s]
            out = part if out is None else out + part
        C = out.shape[1]
        out = (out.reshape(n_i, n_j, C, s, s)
                  .permute(2, 0, 3, 1, 4)
                  .reshape(1, C, n_i * s, n_j * s))
        return out[..., :M, :N]

    def matmul(self, a, b):
        # Dtype-граница: симулятор живёт в fp32/complex64; модель может быть
        # в bf16/fp16 (H100, nanochat). Касты на входе-выходе, градиент через
        # них проходит штатно. Внутренняя точность sim остаётся fp32 — она
        # должна быть лишь ниже шумового пола устройства.
        in_dtype = a.dtype
        if in_dtype in (torch.bfloat16, torch.float16):
            a, b = a.float(), b.float()
        a4, b4 = self._to_4d(a), self._to_4d(b)
        if max(a4.shape[-2], a4.shape[-1], b4.shape[-1]) <= self.tile_size:
            out4 = self._mm(a4, b4)
        else:
            out4 = self._tiled_matmul(a4, b4)
        out = out4[0] if a.dim() == 3 else out4
        return out.to(in_dtype) if out.dtype != in_dtype else out

    def pop_stats(self):
        """Средний и максимальный коэффициент усиления шума с момента прошлого вызова."""
        s, self.stats = self.stats, {}
        n = max(s.get('calls', 0), 1)
        return {'amp_mean': s.get('amp_sum', 0.0) / n,
                'amp_max': s.get('amp_max', 0.0),
                'calls': s.get('calls', 0)}

    @torch.no_grad()
    def calibrate_gain(self, n_trials=8, device=None, dtype=torch.float32):
        """
        Автокалибровка скалярного gain по реперным матрицам ПОЛНОЙ апертуры.
        Внимание: из-за краевых эффектов этот скаляр отличается от масштаба
        малых центральных матриц на десятки процентов — это НОРМАЛЬНО и
        безопасно: точный масштаб на форму дают flat-field карты (в т.ч.
        внутри двойника backward='twin'). Скаляр задаёт лишь единицы полной
        шкалы детектора для шумовой модели; процедура сохранена неизменной
        ради непрерывности единиц сигм между сериями экспериментов.
        """
        was = self.noise
        self.noise = None          # калибруемся на чистом тракте
        num, den = 0.0, 0.0
        for _ in range(n_trials):
            P = torch.rand(1, 1, self.size, self.size, device=device, dtype=dtype)
            Q = torch.rand(1, 1, self.size, self.size, device=device, dtype=dtype)
            raw = self.sim(P, Q)
            ref = torch.matmul(P, Q)
            num += (raw * ref).sum().item()
            den += (raw * raw).sum().item()
        self.noise = was
        self.gain = num / den
        return self.gain


def mm(engine, a, b, optical):
    """Диспетчер одного матмула: оптический движок или обычный torch.matmul."""
    if optical:
        return engine.matmul(a, b)
    return torch.matmul(a, b)


class OpticLinear(nn.Module):
    """
    Линейный слой с переключаемым бэкендом. Вес всегда [in, out], forward считает
    x @ weight (identical parametrization для optical и digital → честное сравнение).
      * optical=True  → x @ weight через engine (оптика);
      * optical=False → обычный torch x @ weight (цифра).
    bias прибавляется снаружи (в честном масштабе, не через sim).

    engine хранится вне реестра подмодулей (в кортеже), чтобы один физический
    симулятор не дублировался в state_dict у каждого слоя; на устройство его
    перемещает регистрация engine на самой модели.
    """
    def __init__(self, in_features, out_features, engine=None, optical=True,
                 bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert not (optical and engine is None), "optical=True требует engine"
        self.in_features = in_features
        self.out_features = out_features
        self.optical = optical
        self._engine = (engine,)
        self.weight = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def engine(self):
        return self._engine[0]

    def reset_parameters(self):
        # Вес [in, out], forward = x @ weight → корректный fan_in = in_features.
        # bound = 1/sqrt(in_features) ≡ kaiming_uniform_(a=sqrt(5)) при правильном fan_in.
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = mm(self.engine, x, self.weight, self.optical)
        return y + self.bias if self.bias is not None else y

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, optical={self.optical}")


class OpticalAttention(nn.Module):
    """
    Каузальное self-attention целиком в обёртке, с двумя независимыми флагами:
      * proj_optical — оптические ли q/k/v/o проекции;
      * attn_optical — оптические ли матмулы q@kᵀ и attn@v.
    Любая их комбинация валидна (напр. цифровые проекции + оптический attention,
    как в одной из твоих веток). rope инжектируется снаружи (или None).

    Порядок как в рабочем коде: proj → split_to_heads → rope → q@kᵀ → mask → softmax
    → attn@v → gather_heads → o_proj. Масштаб только h_dim**-0.5 (без обучаемых k1/k2).
    """
    def __init__(self, h_dim, engine=None, rope=None, num_heads=1,
                 proj_optical=True, attn_optical=True, bias=True, causal=True):
        super().__init__()
        assert not ((proj_optical or attn_optical) and engine is None), \
            "оптический режим требует engine"
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.attn_optical = attn_optical
        self.causal = causal
        self._engine = (engine,)
        self.rope = rope
        self.q_proj = OpticLinear(h_dim, h_dim, engine, optical=proj_optical, bias=bias)
        self.k_proj = OpticLinear(h_dim, h_dim, engine, optical=proj_optical, bias=bias)
        self.v_proj = OpticLinear(h_dim, h_dim, engine, optical=proj_optical, bias=bias)
        self.o_proj = OpticLinear(h_dim, h_dim, engine, optical=proj_optical, bias=bias)
        self._mask = None  # ленивый кеш каузальной маски (растёт до максимального T)

    @property
    def engine(self):
        return self._engine[0]

    def _split(self, x, B, T):
        if self.num_heads <= 1: return x
        return rearrange(x, 'b t (n h) -> (b n) t h', b=B, t=T, n=self.num_heads)

    def _gather(self, x, B, T):
        if self.num_heads <= 1: return x
        return rearrange(x, '(b n) t h -> b t (n h)', b=B, t=T, n=self.num_heads)

    def forward(self, x):
        B, T, _ = x.shape
        q = self._split(self.q_proj(x), B, T)
        k = self._split(self.k_proj(x), B, T)
        v = self._split(self.v_proj(x), B, T)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        head_dim = self.h_dim // self.num_heads
        scores = mm(self.engine, q, k.transpose(-2, -1), self.attn_optical) * (head_dim ** -0.5)
        if self.causal:
            if (self._mask is None or self._mask.size(0) < T
                    or self._mask.device != x.device):
                self._mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                                   device=x.device))
            scores = scores.masked_fill(~self._mask[:T, :T], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = mm(self.engine, attn, v, self.attn_optical)
        return self.o_proj(self._gather(out, B, T))