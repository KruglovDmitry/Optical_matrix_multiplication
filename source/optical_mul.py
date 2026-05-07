import torch as _torch
import torch.nn as _nn
from .config import Config as _Config, LumaiOpticConfig as _LumaiOpticConfig, SummingConfig as _SummingConfig
from .propagator import PropagatorCrossLens as _PropCrossLens, PropagatorСylindLens as _PropСylindLens, PropagatorSinc as _PropSinc, Propagator as _Prop, PropagatorFreeSpaceLumai as _PropagatorFreeSpaceLumai, PropagatorFanOut as _PropagatorFanOut, PropagatorSummingLens as _PropagatorSummingLens

class OpticalMul(_nn.Module):
    """
    Класс системы, выполняющей оптически операцию умножения матрицы на матрицу.
    """
    def __init__(self, config: _Config):
        """
        Конструктор класса.
 
        Args:
            config: конфигурация расчётной системы.
        """
        super(OpticalMul, self).__init__()

        prop_one = _PropSinc(config.input_vector_plane, config.first_lens_plane, config)
        prop_two = _PropCrossLens(config.first_lens_plane, config)
        prop_three = _PropSinc(config.first_lens_plane, config.matrix_plane, config)
        prop_four = _PropСylindLens(config.matrix_plane, config)
        prop_five = _PropSinc(config.matrix_plane, config.second_lens_plane, config)
        prop_six = _PropCrossLens(config.second_lens_plane, config).T
        prop_seven = _PropSinc(config.second_lens_plane, config.output_vector_plane, config)

        self._propagator_one: _Prop = prop_one + prop_two + prop_three + prop_four
        self._propagator_two: _Prop = prop_five + prop_six + prop_seven

        kron_vec_utils = _torch.ones((config.input_vector_split_y, config.input_vector_split_x))
        kron_mat_utils = _torch.ones((config.matrix_split_x, config.matrix_split_y))
        self.register_buffer('_kron_vec_utils', kron_vec_utils, persistent=True)
        self.register_buffer('_kron_mat_utils', kron_mat_utils, persistent=True)
        
        self._avg_pool = _nn.AvgPool2d((1, config.result_vector_split))

    def prepare_vector(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки матрицы левой матрицы, как набора векторов столбцов, к подаче на вход системы.

        Args:
            data: матрица комплексной амплитуды распределений световых полей.

        Returns:
            Матрицы содержащие вектора левой матрицы.
        """
        data = data.cfloat().flip(-1)
        data = data.unsqueeze(-2)
        data = _torch.kron(data.contiguous(), self._kron_vec_utils)
        return data

    def prepare_matrix(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Метод подготовки правой матрицы к подаче на вход системы.

        Args:
            data: матрица комплексной амплитуды распределения светового поля.

        Returns:
            Матрица - оптический элемент в центре модели.
        """
        if (data.dim() > 4) and data.size(-1) == 2:
            data = _torch.view_as_complex(data)

        data = data.cfloat().transpose(-2, -1)
        data = data.unsqueeze(-3)
        data = _torch.kron(data.contiguous(), self._kron_mat_utils)
        return data

    def prepare_out(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Метод получения результата матричного умножения.

        Args:
            data: матрицы выходого распределения светового поля системы.

        Returns:
            Вектор столбец (амплитудное распределение).
        """
        ### Закоментированная часть кода - более физически корректный вариант работы модели,
        ### однако, данный вариант кода будет требовать большое кол-во памяти во время обучения
        field = field.abs().squeeze(-1) #**2
        field = self._avg_pool(field)
        return field.flip(-1) #**0.5

    def forward(self,
                input: _torch.Tensor,
                other: _torch.Tensor) -> _torch.Tensor:
        """
        Метод выполения матричного умножения.

        Args:
            input: матрица (B, C, H, W).
            other: матрица (B, C, W, K).

        Returns:
            Рензультат матричного умножения (B, C, H, K).

        Example:
            >>> mul = OpticalMul(...)
            >>> A = torch.rand((1, 1, 256, 256)) > 0.5
            >>> B = torch.rand((1, 1, 256, 256)) > 0.5
            >>> mul(A, B).shape
            torch.Size([1, 1, 256, 256])
            >>> A = torch.rand((1, 1, 64, 256)) > 0.5
            >>> B = torch.rand((1, 1, 256, 128)) > 0.5
            >>> mul(A, B).shape
            torch.Size([1, 1, 64, 128])
        """
        vec_field = self.prepare_vector(input)
        mat_field = self.prepare_matrix(other)

        vec_field = self._propagator_one(vec_field, mat_field.shape[-2:])
        vec_field = self._propagator_two(vec_field * mat_field, (mat_field.size(-2), 1))

        return self.prepare_out(vec_field)
    
class LumaiMul(_nn.Module):
    """
    Оптическое матрично-векторное умножение в архитектуре Lumai.
 
    Физический pipeline одного оптического такта:
 
        [M VCSEL лазеров]
             │  амплитуда x[i] пропорциональна элементу вектора
             ↓
        [PropagatorFreeSpaceLumai]
             │  sinc-распространение: лучи расходятся от лазеров к дисплею
             ↓
        [PropagatorFanOut]
             │  гауссов fan-out: каждый луч i покрывает все N столбцов дисплея
             ↓
        [Плоскость дисплея весов W[M, N]]
             │  Адамарово умножение: field[i,j] *= W[i,j]
             │  (некогерентный свет → только амплитудная модуляция)
             ↓
        [PropagatorSummingLens]
             │  физическое суммирование: все M лучей столбца j → детектор j
             ↓
        [PropagatorFreeSpaceLumai]
             │  sinc-распространение: от дисплея до детектора
             ↓
        [N детекторов]
             │  y[j] = Σᵢ |x[i] · W[i,j]|²  (некогерентно)
             │        или |Σᵢ x[i] · W[i,j]|² (когерентно)
 
    Сравнение с OpticalMul (4f/POMMM):
    ┌──────────────────┬─────────────────────────┬──────────────────────────┐
    │                  │ OpticalMul (4f)          │ LumaiMul                 │
    ├──────────────────┼─────────────────────────┼──────────────────────────┤
    │ Операция         │ MMM (матрица × матрица)  │ MVM (матрица × вектор)   │
    │ Источник         │ Когерентный лазер 532нм  │ VCSEL матрица 850нм      │
    │ Линза 1          │ CrossLens (Фурье)        │ FanOut (рассеивающая)    │
    │ Центр            │ Цилиндрическая линза     │ Дисплей весов (некогер.) │
    │ Линза 2          │ CrossLens.T (Фурье)      │ SummingLens (собирающая) │
    │ Суммирование     │ Обратный FFT (оператор)  │ Физическое (оператор)    │
    │ Детектор         │ Амплитуда |E|            │ Интенсивность Σ|E_i|²    │
    │ Масштаб          │ ~50×50 (эксперимент)     │ 1024×2048 (заявлено)     │
    │ Пропагаторов     │ 7                        │ 4 (схлопываются в 2)     │
    └──────────────────┴─────────────────────────┴──────────────────────────┘
 
    Совместимость с OpticalMul:
        Тот же интерфейс forward(input, other) → результат.
        input:  (B, C, H, M) — левая матрица (батч строк-векторов)
        other:  (B, C, M, N) — правая матрица (матрица весов дисплея)
        output: (B, C, H, N) — результат умножения
 
    Args:
        config: LumaiOpticConfig с физическими параметрами установки.
    """
    def __init__(self, config: _LumaiOpticConfig):
        super().__init__()
 
        self._config = config
 
        # ── Строим операторы распространения ──────────────────────────────
 
        # Конфиг для второго этапа (дисплей → детектор)
        # Используем тот же базовый конфиг но с summing_distance
        config_summing = _SummingConfig(config)
 
        # Этап 1: от лазеров до дисплея весов
        #   prop_fs_in:  sinc-распространение (лазер → дисплей)
        #   prop_fanout: fan-out линза (broadcast по столбцам)
        prop_fs_in = _PropagatorFreeSpaceLumai(
            config.laser_plane,
            config.display_plane,
            config
        )
        prop_fanout = _PropagatorFanOut(
            config.laser_plane,
            config.display_plane,
            config
        )
 
        # Схлопываем в один оператор: field → display_plane
        # (как prop_one + prop_two + prop_three в OpticalMul)
        self._propagator_one: _Prop = prop_fs_in + prop_fanout
 
        # Этап 2: от дисплея весов до детекторов
        #   prop_summing: суммирующая линза
        #   prop_fs_out: sinc-распространение (дисплей → детектор)
        prop_summing = _PropagatorSummingLens(
            config.display_plane,
            config.detector_plane,
            config
        )
        prop_fs_out = _PropagatorFreeSpaceLumai(
            config.display_plane,
            config.detector_plane,
            config_summing
        )
 
        # Схлопываем в один оператор: display_plane → детектор
        # (как prop_five + prop_six + prop_seven в OpticalMul)
        self._propagator_two: _Prop = prop_summing + prop_fs_out
 
        # Квантование дисплея весов (STE для обучения)
        self._display_bits = config.display_bits
        self._display_levels = 2 ** config.display_bits - 1
 
        # Некогерентный или когерентный режим детектора
        self._incoherent = config.incoherent
 
        # AvgPool для субпиксельного усреднения (аналог result_vector_split)
        self._avg_pool = _nn.AvgPool2d((1, 1))
 
    def prepare_vector(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Подготовка левой матрицы как набора входных векторов.
 
        В Lumai входной вектор x[i] кодируется интенсивностью i-го лазера.
        Лазеры расположены в 1D, поэтому поле имеет форму [H, M, 1] —
        H независимых векторов, каждый из M элементов, в одном пространственном
        измерении (ось Y = лазеры, ось X = 1 точка на лазер).
 
        Args:
            data: (B, C, H, M) — левая матрица
 
        Returns:
            (B, C, H, M, 1) — каждый вектор как колонка поля
        """
        # Берём abs() потому что Lumai использует некогерентный свет —
        # интенсивность лазера (не фазу) кодирует значение.
        # Sqrt потому что детектор меряет |E|², а мы хотим чтобы
        # амплитуда E = sqrt(intensity) давала правильную интенсивность.
        data = data.abs().to(_torch.cfloat)
        return data.unsqueeze(-1)  # (B, C, H, M, 1)
 
    def prepare_matrix(self, data: _torch.Tensor) -> _torch.Tensor:
        """
        Подготовка правой матрицы как матрицы весов дисплея.
 
        Дисплей Lumai — амплитудный модулятор. Каждый пиксель W[i,j] ∈ [0,1]
        задаёт пропускание для луча i на детектор j.
        В отличие от 4f системы здесь нет транспонирования и kron-расширения —
        дисплей физически расположен в плоскости [M, N].
 
        Квантование до display_bits уровней (физическое ограничение дисплея).
        Straight-through estimator сохраняет градиент для обучения.
 
        Args:
            data: (B, C, M, N) — правая матрица
 
        Returns:
            (B, C, 1, M, N) — готово к Адамарову умножению
        """
        if data.dim() > 4 and data.size(-1) == 2:
            data = _torch.view_as_complex(data)
 
        # Некогерентный дисплей: только амплитудная модуляция [0, 1]
        data = data.abs().clamp(0.0, 1.0).to(_torch.cfloat)
 
        # Квантование дисплея (физическое ограничение битности)
        if self._display_bits is not None:
            levels = self._display_levels
            data_q = _torch.round(data.real * levels) / levels
            data_q = data_q.to(_torch.cfloat)
            # Straight-through estimator: градиент проходит напрямую
            data = data + (data_q - data).detach()
 
        # Добавляем dim для broadcast с (B, C, H, M, 1) → (B, C, H, M, N)
        return data.unsqueeze(-3)  # (B, C, 1, M, N)
 
    def prepare_out(self, field: _torch.Tensor) -> _torch.Tensor:
        """
        Считывание результата с детектора.
 
        Ключевое отличие от OpticalMul:
        В 4f системе детектор меряет амплитуду поля.
        В Lumai детектор (фотодиод) меряет интенсивность — |E|².
 
        При некогерентном суммировании:
            y[j] = Σᵢ |E[i,j]|² = Σᵢ (x[i] · W[i,j])²
 
        При когерентном суммировании (для сравнения):
            y[j] = |Σᵢ E[i,j]|²
 
        Переход от интенсивности к числовому масштабу x@W:
            sqrt(y[j]) ≈ Σᵢ x[i] · W[i,j]  при малых ошибках
 
        Args:
            field: (B, C, H, 1, N) — выходное поле после суммирующей линзы
 
        Returns:
            (B, C, H, N) — результат умножения
        """
        if self._incoherent:
            # Некогерентный детектор: суммируем интенсивности
            # field здесь уже прошёл через суммирующий оператор,
            # поэтому это суммарная амплитуда — берём |E|²
            result = field.abs() ** 2
        else:
            # Когерентный режим (для сравнения с 4f): амплитуда
            result = field.abs()
 
        # Убираем лишнее измерение суммирования
        result = result.squeeze(-2)  # (B, C, H, N)
 
        # Нормируем к масштабу матричного умножения
        # sqrt для некогерентного режима: |E|² → |E| ~ x@W
        if self._incoherent:
            result = result.sqrt()
 
        return self._avg_pool(result)
 
    def forward(self,
                input: _torch.Tensor,
                other: _torch.Tensor) -> _torch.Tensor:
        """
        Оптическое умножение input @ other в архитектуре Lumai.
 
        Pipeline:
            1. prepare_vector:    (B,C,H,M) → (B,C,H,M,1)
            2. propagator_one:    (B,C,H,M,1) → (B,C,H,M,N)   [fan-out]
            3. prepare_matrix:    (B,C,M,N) → (B,C,1,M,N)
            4. Адамар:            field * weights → (B,C,H,M,N) [дисплей]
            5. propagator_two:    (B,C,H,M,N) → (B,C,H,1,N)   [суммирование]
            6. prepare_out:       (B,C,H,1,N) → (B,C,H,N)
 
        Args:
            input: (B, C, H, M) — левая матрица
            other: (B, C, M, N) — правая матрица (матрица весов дисплея)
 
        Returns:
            (B, C, H, N) — результат умножения
 
        Example:
            >>> cfg = LumaiOpticConfig(n_lasers=64, n_outputs=64)
            >>> mul = LumaiMul(cfg)
            >>> A = torch.rand(1, 1, 32, 64)
            >>> B = torch.rand(1, 1, 64, 64)
            >>> mul(A, B).shape
            torch.Size([1, 1, 32, 64])
        """
        # Шаг 1: подготовка входных данных
        vec_field = self.prepare_vector(input)   # (B,C,H,M,1)
        mat_field = self.prepare_matrix(other)   # (B,C,1,M,N)
 
        # Шаг 2: распространение от лазеров до дисплея (fan-out)
        # Аналог: self._propagator_one(vec_field, mat_field.shape[-2:])
        # operator_Y[M,M] @ field[M,1] @ operator_X[1,N] → field[M,N]
        vec_field = self._propagator_one(
            vec_field,
            mat_field.shape[-2:]   # целевой размер: (M, N)
        )  # (B,C,H,M,N)
 
        # Шаг 3: Адамарово умножение на матрицу весов дисплея
        # Точно как в OpticalMul: vec_field * mat_field
        # mat_field broadcast по H: (B,C,1,M,N) → (B,C,H,M,N)
        vec_field = vec_field * mat_field  # (B,C,H,M,N)
 
        # Шаг 4: суммирующая линза + распространение до детектора
        # operator_Y[1,M] @ field[M,N] @ operator_X[N,N] → field[1,N]
        vec_field = self._propagator_two(
            vec_field,
            (1, mat_field.size(-1))  # целевой размер: (1, N)
        )  # (B,C,H,1,N)
 
        # Шаг 5: считываем результат с детектора
        return self.prepare_out(vec_field)  # (B,C,H,N)
 
class LumaiMulBlocked(_nn.Module):
    """
    Блочное оптическое умножение для матриц с K > n_lasers.
 
    В Lumai n_lasers = 1024. Для умножения матриц с K > 1024
    (например, для слоя трансформера 4096×4096)
    матрица разбивается на блоки по оси K и результаты суммируются.
 
    Это физически корректно: это ровно то что делает реальная установка Lumai
    при обработке матрицы 2048×2048 — два последовательных оптических такта.
 
    При некогерентном свете суммирование блоков = суммирование интенсивностей:
        y[j] = Σ_blocks Σᵢ∈block |x[i] · W[i,j]|²
 
    Args:
        config:     LumaiOpticConfig
        block_size: размер блока по K (по умолчанию = n_lasers)
    """
    def __init__(self, config: _LumaiOpticConfig, block_size: int = None):
        super().__init__()
        self._core = LumaiMul(config)
        self._block_size = block_size or config.n_lasers
        self._incoherent = config.incoherent
 
    def forward(self,
                input: _torch.Tensor,
                other: _torch.Tensor) -> _torch.Tensor:
        """
        Args:
            input: (B, C, H, K)
            other: (B, C, K, N)
 
        Returns:
            (B, C, H, N)
        """
        K = input.shape[-1]
        bs = self._block_size
 
        if K <= bs:
            return self._core(input, other)
 
        result = None
        for k0 in range(0, K, bs):
            k1 = min(k0 + bs, K)
            block = self._core(
                input[..., k0:k1],   # (B,C,H,block)
                other[..., k0:k1, :] # (B,C,block,N)
            )  # (B,C,H,N)
 
            if result is None:
                result = block
            else:
                # Некогерентное сложение: суммируем интенсивности блоков
                # (перед sqrt в prepare_out результат уже ~ sqrt(интенсивности))
                # Поэтому пересчитываем через квадрат, суммируем, берём sqrt
                if self._incoherent:
                    result = (result**2 + block**2).sqrt()
                else:
                    result = result + block
 
        return result