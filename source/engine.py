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
  * 'optical'  — полный офлоад: градиентные произведения тоже через оптику
                (знаковый дизеринг обязателен). Цена +11–15% ppl (реалистичный
                шум: больше); опция для случаев, когда цифра недоступна.

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


def _reduce_to_shape(g, shape):
    """Свернуть broadcast-оси градиента к форме исходного тензора."""
    while g.dim() > len(shape):
        g = g.sum(0)
    for i, (gs, ss) in enumerate(zip(g.shape, shape)):
        if ss == 1 and gs != 1:
            g = g.sum(i, keepdim=True)
    return g


def _dithered_value_mm(eng, A, B):
    """
    Оптическое произведение со знаковым дизерингом: случайные ±1 по строкам A
    и столбцам B коммутируют с матмулом (флипы снимаются в цифре после
    измерения), но полностью меняют реализацию сдвигов/нормировок каждый
    вызов. Детерминированная входозависимая неидеальность тракта из
    систематического bias градиента (фактор ~0.65 на живых градиентах)
    превращается в средненулевой шум между шагами, который SGD усредняет.
    Флипы по строкам/столбцам не пересекают операнды — broadcast-формы
    не расширяются, лишних проходов нет.
    """
    r2 = lambda *sh: (torch.randint(0, 2, sh, device=A.device) * 2 - 1).to(A.dtype)
    sr = r2(*A.shape[:-2], A.shape[-2], 1)   # знаки строк A -> строки результата
    sc = r2(*B.shape[:-2], 1, B.shape[-1])   # знаки столбцов B -> столбцы результата
    return eng.value_matmul(A * sr, B * sc) * sr * sc


class _OpticalGemmGrad(torch.autograd.Function):
    """
    Ref-узел PAT-S с оптическим backward (полный офлоад обучения).

    Forward возвращает НУЛИ: значение ref сокращается в straight-through
    (raw = ref + (raw_phys − ref).detach() даёт raw_phys при любом значении
    ref), поэтому цифровой матмул в forward не нужен вовсе.

    Backward считает оба градиентных произведения (G@Qnᵀ и Pnᵀ@G) через
    (шумную) оптику: градиенты — это просто матмулы, дифференцируемость
    сквозь них не требуется. Итог: 3 оптических прохода, 0 больших цифровых
    матмулов — так работают и мемристорные кроссбары. Вопрос эксперимента:
    терпит ли обучение шум устройства в самих градиентах.
    """
    @staticmethod
    def forward(ctx, Pn, Qn, engine):
        ctx.save_for_backward(Pn, Qn)
        ctx.engine = engine
        shape = (torch.broadcast_shapes(Pn.shape[:-2], Qn.shape[:-2])
                 + (Pn.shape[-2], Qn.shape[-1]))
        return Pn.new_zeros(shape)

    @staticmethod
    def backward(ctx, grad_out):
        Pn, Qn = ctx.saved_tensors
        eng = ctx.engine
        mm = (_dithered_value_mm if getattr(eng, 'ob_dither', True)
              else lambda e, a, b: e.value_matmul(a, b))
        with torch.no_grad():
            dP = mm(eng, grad_out, Qn.transpose(-2, -1))
            dQ = mm(eng, Pn.transpose(-2, -1), grad_out)
        return (_reduce_to_shape(dP, Pn.shape),
                _reduce_to_shape(dQ, Qn.shape), None)


def optics_matmul(sim, A, B, eps=1e-8, gain=1.0, noise=None, stats=None,
                  ffmap=None, sim_st=False, bwd_engine=None):
    """
    Приближённо вычисляет A @ B через неотрицательный оптический sim.
    A: [..., M, K], B: [..., K, N] — уже 4D с согласованными (или broadcast)
    ведущими осями (об этом заботится OpticalEngine).

    Оптика принимает только неотрицательные интенсивности, поэтому:
      1) строки A и столбцы B сдвигаются в неотрицательную область (sa, sb);
      2) сдвинутые P, Q нормируются в [0,1] (делением на построчный/постолбцовый max);
      3) sim перемножает нормированные матрицы, масштаб восстанавливается (* a * b * gain);
      4) три поправки за сдвиг вычитаются аналитически (в честном масштабе — gain на них НЕ идёт).
    Раскрытие (A+sa)(B+sb) даёт ровно эти поправки — результат равен A @ B.

    noise — OpticalNoiseModel или None. Шум модуляции ложится на нормированные
    входы sim, шум детектора — на сырой выход ДО * a * b * gain, поэтому его
    вклад в результат усиливается пропорционально масштабу P@Q (то самое
    катастрофическое сокращение при вычитании поправок).

    stats — mutable dict для аудита усиления шума: накапливает отношение
    ||PQ|| / ||A@B|| (во сколько раз аддитивный шум детектора раздувается
    относительно полезного сигнала).
    """
    k = A.shape[-1]
    sa = torch.clamp(-A.amin(dim=-1, keepdim=True), min=0)   # [...,M,1]
    sb = torch.clamp(-B.amin(dim=-2, keepdim=True), min=0)   # [...,1,N]
    P, Q = A + sa, B + sb
    a = P.amax(dim=-1, keepdim=True).clamp_min(eps)
    b = Q.amax(dim=-2, keepdim=True).clamp_min(eps)
    Pn, Qn = P / a, Q / b
    if sim_st:
        # PAT-S («хирургический»): straight-through ТОЛЬКО вокруг физического
        # блока sim+детектор. Сдвиги, нормировки a·b и поправки — цифровые
        # шаги хоста и на реальном устройстве, они остаются в autograd с
        # РЕАЛИЗОВАННЫМИ значениями. Невязка δ = (измерено − предсказано)
        # наблюдаема на железе; реализация шума протекает в градиент через
        # член raw_phys·∂(a·b)/∂A — главный канал адаптации при полном
        # autograd. Приближается только якобиан внутренностей sim (цена
        # ~+1.6% по P_baseline). Sim под no_grad — активации не копятся.
        with torch.no_grad():
            Pm = noise.modulate(Pn) if noise is not None else Pn
            Qm = noise.modulate(Qn) if noise is not None else Qn
            raw_phys = sim(Pm, Qm)
            if noise is not None:
                raw_phys = noise.detect(raw_phys, full_scale=k / gain)
        if bwd_engine is not None:
            # Оптический backward: forward-значение ref — нули (сократится),
            # градиентные произведения посчитает оптика в Function.backward.
            ref = _OpticalGemmGrad.apply(Pn, Qn, bwd_engine) / gain
        else:
            ref = torch.matmul(Pn, Qn) / gain
        if ffmap is not None:
            # КРИТИЧНО: двойник обязан включать калибровку устройства.
            # raw_phys ≈ matmul/(gain·ffmap), и ref должен жить в том же
            # масштабе, иначе δ содержит большую детерминированную часть
            # (в т.ч. скалярную ошибку gain — например, calibrate_gain на
            # полной апертуре даёт до ~40% сдвига для малых матриц), которая
            # ломает сокращение нормировочных градиентов и медленно сносит
            # обучение. С картой в ref δ ≈ шум + малый входозависимый остаток.
            ref = ref / ffmap.clamp_min(1e-12)
        raw = ref + (raw_phys - ref).detach()
    else:
        if noise is not None:
            Pn, Qn = noise.modulate(Pn), noise.modulate(Qn)
        raw = sim(Pn, Qn)                   # сырые интенсивности детектора
        if noise is not None:
            raw = noise.detect(raw, full_scale=k / gain)
    if ffmap is not None:
        # Flat-field: попиксельная коррекция фиксированной неоднородности
        # усиления. Применяется ПОСЛЕ детектора/АЦП — как в цифровом
        # постпроцессинге реального устройства.
        raw = raw * ffmap
    PQ = raw * a * b * gain
    corr_a = sb * A.sum(dim=-1, keepdim=True)
    corr_b = sa * B.sum(dim=-2, keepdim=True)
    corr_c = k * sa * sb
    out = PQ - corr_a - corr_b - corr_c
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
                 ff_trials=8, backward='twin', ob_dither=True, tile_batch=None):
        super().__init__()
        assert backward in ('twin', 'autograd', 'optical'), backward
        self.size = size
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
        # Знаковый дизеринг оптического backward. ОБЯЗАТЕЛЕН на практике:
        # без него детерминированная неидеальность тракта даёт систематический
        # bias градиента (фактор ~0.65) и плато (11.2 vs 6.06 ppl).
        # Флаг оставлен только для абляций.
        self.ob_dither = ob_dither
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
                   distance=distance)
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
        # Двойник нужен только когда строится граф; на инференсе (no_grad)
        # и в режиме autograd идём обычной веткой — цифровой матмул не считается.
        need_twin = self.backward != 'autograd' and torch.is_grad_enabled() and (a4.requires_grad or b4.requires_grad)
        return optics_matmul(
            self.sim, a4, b4, gain=self.gain, noise=self.noise,
            stats=stats, ffmap=ffmap, sim_st=need_twin,
            bwd_engine=self if (need_twin and self.backward == 'optical') else None)

    def _tiled_sequential(self, a4, b4):
        s = self.size
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
        s = self.size
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
        out = None
        for kk in range(n_k):
            acc_k = []
            for p0 in range(0, P, self.tile_batch):
                sl = slice(p0, min(p0 + self.tile_batch, P))
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
        a4, b4 = self._to_4d(a), self._to_4d(b)
        if max(a4.shape[-2], a4.shape[-1], b4.shape[-1]) <= self.size:
            out4 = self._mm(a4, b4)
        else:
            out4 = self._tiled_matmul(a4, b4)
        return out4[0] if a.dim() == 3 else out4

    @torch.no_grad()
    def value_matmul(self, a, b):
        """Оптическое значение произведения без ST/двойника (для backward
        и любых мест, где нужен только результат). Временное переключение
        режима не потокобезопасно — исследовательский код."""
        save = self.backward
        self.backward = 'autograd'
        try:
            return self.matmul(a, b)
        finally:
            self.backward = save

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
        scores = mm(self.engine, q, k.transpose(-2, -1), self.attn_optical) * (self.h_dim ** -0.5)
        if self.causal:
            if (self._mask is None or self._mask.size(0) < T
                    or self._mask.device != x.device):
                self._mask = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                                   device=x.device))
            scores = scores.masked_fill(~self._mask[:T, :T], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = mm(self.engine, attn, v, self.attn_optical)
        return self.o_proj(self._gather(out, B, T))