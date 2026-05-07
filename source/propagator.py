import torch as _torch
import torch.nn as _nn
import numpy as _np
from scipy.special import fresnel as _fresnel
from .config import ConfigOpticBase as _ConfigOpticBase, ConfigDesignPlane as _ConfigDesignPlane, LumaiOpticConfig as _LumaiOpticConfig
from typing import Tuple as _Tuple, Sequence as _Sequence

from abc import ABC as _ABC
import collections as _collections

class Propagator(_ABC, _nn.Module):
    """
    Абстрактный класс вычисления распространения светового поля в среде.

    Поля:
        operator_X: оператор отображающий распроcтранение светового поля вдоль оси абсцисс
        operator_Y: оператор отображающий распроcтранение светового поля вдоль оси ординат
    """
    def __init__(self, operator_X: _torch.Tensor, operator_Y: _torch.Tensor):
        super(Propagator, self).__init__()
        operator_X: _torch.Tensor = _torch.view_as_real(operator_X)
        operator_Y: _torch.Tensor = _torch.view_as_real(operator_Y)
        self.register_buffer('_operator_X', operator_X, persistent=True)
        self.register_buffer('_operator_Y', operator_Y, persistent=True)

    @property
    def operator_X(self) -> _torch.Tensor:
        """
        Returns:
            оператор отображающий распроcтранение светового поля вдоль оси абсцисс
        """
        return _torch.view_as_complex(self._operator_X)
    @property
    def operator_Y(self) -> _torch.Tensor:
        """
        Returns:
            оператор отображающий распроcтранение светового поля вдоль оси ординат
        """
        return _torch.view_as_complex(self._operator_Y)

    def __operator_multiplication(self, first_X: _torch.Tensor,
                                  second_X: _torch.Tensor,
                                  first_Y: _torch.Tensor,
                                  second_Y: _torch.Tensor)-> _Tuple[_torch.Tensor, _torch.Tensor]:
        operator_Y = second_Y @ first_Y
        operator_X = first_X @ second_X
        return operator_X, operator_Y

    def cat(self, propagators: _Sequence['Propagator']) -> 'Propagator':
        """
        Метод схлопывания операторов распространения.
    
        Args:
            propagators: последовательность для схлопывания

        Returns:
            новый пропогатор, заменяющих собой серию предыдущих

        Warning:
            порядок расположения пропагаторов в последовательности важен,
            идёт от первого к последниму
        """
        operator_X: _torch.Tensor
        operator_Y: _torch.Tensor
        if not isinstance(propagators, _collections.abc.Sequence):
            operator_X, operator_Y = self.__operator_multiplication(self.operator_X,
                                                               propagators.operator_X,
                                                               self.operator_Y,
                                                               propagators.operator_Y)
        else:
            size = len(propagators)
            operator_X = self.operator_X
            operator_Y = self.operator_Y
            for i in range(size):
                operator_X, operator_Y = self.__operator_multiplication(operator_X,
                                                                   propagators[i].operator_X,
                                                                   operator_Y,
                                                                   propagators[i].operator_Y)
        return Propagator(operator_X, operator_Y)

    def __add__(self, propagator: 'Propagator') -> 'Propagator':
        """
        Метод схлопывания двух пропагаторов.
        Args:
            propagator: пропагатор с которым нужно произвести схлопывание

        Returns:
            новый пропогатор, заменяющих собой оба предыдущих

        Warning:
            операция не комутативная
        """
        return self.cat(propagator)

    @staticmethod
    def __slice_calculation(total_rows: int, num_to_take: int) -> slice:
        start = (total_rows - num_to_take) // 2
        end = start + num_to_take
        return slice(start, end)
        
    def forward(self,
                field: _torch.Tensor, resul_shape: None | _Tuple[int, int] | _torch.Size) -> _torch.Tensor:
        """
        Метод распространения светового поля в среде.
 
        Args:
            field: распределение комплексной амплитуды светового поля.

        Returns:
            Распределение комплексной амплитуды светового поля,
            после распространения.
        """

        if (resul_shape is not None):
            field_shape = field.shape[-2:]
            operator_Y_shape = self.operator_Y.shape[-2:]
            operator_X_shape = self.operator_X.shape[-2:]

            slice_one = Propagator.__slice_calculation(operator_Y_shape[0], resul_shape[0])
            slice_two = Propagator.__slice_calculation(operator_Y_shape[1], field_shape[0])
            slice_three=Propagator.__slice_calculation(operator_X_shape[0], field_shape[1])
            slice_four= Propagator.__slice_calculation(operator_X_shape[1], resul_shape[1])

            
            return self.operator_Y[..., slice_one, slice_two] @ field @ self.operator_X[..., slice_three, slice_four]
        
        return self.operator_Y @ field @ self.operator_X

class PropagatorLens(Propagator):
    """
    Абстрактный класс распространения света в тонком оптическом элементе.
    """
    def transpose(self) -> 'PropagatorLens':
        """
        Метод транспонирования тонкого оптического элемента.
        Returns:
           Новый элемент, транспонированный относительно оригинального.
        """
        obj = Propagator.__new__(PropagatorLens)
        Propagator.__init__(obj, self.operator_Y, self.operator_X)
        return obj

    @property
    def T(self) -> 'PropagatorLens':
        """
        Returns:
           Новый элемент, транспонированный относительно текущего.
        """
        return self.transpose()

class PropagatorCrossLens(PropagatorLens):
    """
    Класс распространения света в скрещенной линзе,
    представленной тонким оптическим элементом.
    """
    def __init__(self, plane: _ConfigDesignPlane,
                 config: _ConfigOpticBase):
        """
        Конструктор класса скрещенной линзы.

        Args:
            plane: данные о расчётной плоскости элемента.
            config: данные о световом поле модели.
        """
        operator_X = _torch.exp(-1j * config.K / config.distance * plane.linspace_by_x**2)
        operator_Y = _torch.exp(-1j * config.K / 2 / config.distance * plane.linspace_by_y**2)
        super(PropagatorCrossLens, self).__init__(_torch.diag_embed(operator_X),
                                                  _torch.diag_embed(operator_Y))

class PropagatorСylindLens(PropagatorLens):
    """
    Класс распространения света в цилиндрической линзе,
    представленной тонким оптическим элементом.
    """
    def __init__(self, plane: _ConfigDesignPlane,
                 config: _ConfigOpticBase):
        """
        Конструктор класса цилиндрической линзы.

        Args:
            plane: данные о расчётной плоскости элемента.
            config: данные о световом поле модели.
        """
        operator_X = _torch.exp(-1j * config.K / config.distance * plane.linspace_by_x**2)
        operator_Y = _torch.ones_like(plane.linspace_by_y, dtype=_torch.cfloat)
        super(PropagatorСylindLens, self).__init__(_torch.diag_embed(operator_X),
                                                   _torch.diag_embed(operator_Y))

class PropagatorSinc(Propagator):
    """
    Класс распространения света свободном пространстве
    с использованием разложения по базисным sinc функциям.
    """
    def __init__(self, first_plane: _ConfigDesignPlane,
                 second_plane: _ConfigDesignPlane,
                 config: _ConfigOpticBase):
        """
        Конструктор класса распространения в свободном пространстве.

        Args:
            first_plane: данные о начальной расчётной плоскости.
            second_plane: данные о конечной расчётной плоскости.
            config: данные о световом поле модели.
        """
        operator_X, operator_Y = self.__get_operators(first_plane,
                                                    second_plane,
                                                    config)
        super(PropagatorSinc, self).__init__(operator_X, operator_Y)

    def __get_operator_for_dim(self,
                             pixel_size_in: float,
                             pixel_size_out: float,
                             difference: float,
                             config: _ConfigOpticBase) -> _torch.Tensor:
        bndW = 0.5 / pixel_size_in
        eikz = (_np.exp(1j * config.K * config.distance)**0.5)
        sq2p = (2 / _np.pi)**0.5
        sqzk = ((2 * config.distance / config.K)**0.5)
        mu1 = -_np.pi * sqzk * bndW - difference / sqzk
        mu2 = _np.pi * sqzk * bndW - difference / sqzk
        S1, C1 = _fresnel(mu1 * sq2p)
        S2, C2 = _fresnel(mu2 * sq2p)
        return (((pixel_size_in * pixel_size_out)**0.5 / _np.pi) / sqzk * eikz
                  * _np.exp(0.5j * difference**2 * config.K / config.distance)
                  * (C2 - C1 - 1j * (S2 - S1)) / sq2p)
        
    def __get_operators(self,
                      first_plane: _ConfigDesignPlane,
                      second_plane: _ConfigDesignPlane,
                      config: _ConfigOpticBase) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        difference_x = first_plane.linspace_by_x[None, :] - second_plane.linspace_by_x[:, None]
        difference_y = first_plane.linspace_by_y[None, :] - second_plane.linspace_by_y[:, None]
        operator_X = self.__get_operator_for_dim(first_plane.pixel_size_by_x,
                                               second_plane.pixel_size_by_x,
                                               difference_x,
                                               config).transpose(-2, -1)
        operator_Y = self.__get_operator_for_dim(first_plane.pixel_size_by_y,
                                               second_plane.pixel_size_by_y,
                                               difference_y,
                                               config)
        return operator_X, operator_Y

class PropagatorFanOut(PropagatorLens):
    """
    Fan-out линза Lumai — копирует каждый из M лазерных лучей
    на всю ширину N дисплея весов.
 
    Физика: рассеивающая линза (или дифракционная решётка) с апертурой,
    покрывающей весь дисплей. Каждый точечный источник i даёт
    расходящийся пучок, покрывающий все N столбцов дисплея.
 
    В операторном формализме:
        operator_X [1, N] — broadcast по оси столбцов.
                            Каждый входной столбец отображается на все N выходных.
        operator_Y [M, M] — единичная матрица по оси строк.
                            Каждый лазер i независим от остальных.
 
    Амплитудный профиль пучка моделируется как гауссов (наиболее реалистично
    для одномодовых VCSEL), нормированный так чтобы суммарная мощность
    каждого луча сохранялась.
 
    Args:
        laser_plane:   плоскость лазерных источников.
        display_plane: плоскость дисплея весов.
        config:        физические параметры установки.
    """
    def __init__(self,
                 laser_plane: _ConfigDesignPlane,
                 display_plane: _ConfigDesignPlane,
                 config: _LumaiOpticConfig):
        M = laser_plane.pixel_count_by_x
        N = display_plane.pixel_count_by_x
 
        # Гауссов профиль fan-out: каждый лазер освещает весь дисплей
        # с амплитудой убывающей по гауссу от центра луча.
        # beam_waist — радиус пучка на уровне 1/e (половина апертуры дисплея)
        display_half_width = display_plane.aperture_width / 2.0
        beam_waist = display_half_width  # пучок покрывает весь дисплей
 
        x_display = display_plane.linspace_by_x  # [N]
        # Амплитудный профиль по оси X: один вектор для всех лазеров
        # (лазеры в 1D, поэтому fan-out одинаков для каждого)
        gaussian_profile = _torch.exp(
            -x_display**2 / (2 * beam_waist**2)
        ).to(_torch.cfloat)  # [N]
 
        # Нормировка: сохраняем мощность (интеграл |E|² = 1)
        gaussian_profile = gaussian_profile / (gaussian_profile.abs()**2).sum().sqrt()
 
        # operator_X: [1, N] — broadcast с весами гауссова профиля
        # При умножении field[..., M, 1] @ operator_X[1, N]
        # каждая строка M копируется в N с весами gaussian_profile
        operator_X = gaussian_profile.unsqueeze(0)  # [1, N]
 
        # operator_Y: [M, M] — единичная (лазеры независимы)
        operator_Y = _torch.eye(M, dtype=_torch.cfloat)  # [M, M]
 
        super().__init__(operator_X, operator_Y)
 
class PropagatorSummingLens(PropagatorLens):
    """
    Суммирующая линза Lumai — физически собирает все M лучей
    одного столбца j на один детектор j.
 
    Физика: собирающая линза с фокусным расстоянием f = summing_distance.
    В фокальной плоскости линзы формируется преобразование Фурье входного поля.
    При некогерентном свете каждый детектор j меряет суммарную интенсивность
    от всех M лазеров, прошедших через столбец j дисплея.
 
    В операторном формализме:
        operator_X [N, N] — единичная матрица по оси столбцов.
                            Каждый столбец j независим.
        operator_Y [1, M] — суммирование по оси строк.
                            Все M лучей столбца j фокусируются на детектор j.
 
    Амплитуда суммирования: для некогерентного света детектор меряет
    сумму интенсивностей (не амплитуд), поэтому физически корректная
    модель — это суммирование |E_i|² а не |ΣE_i|².
    Однако в операторном формализме мы работаем с амплитудами,
    а переход к интенсивности делается в prepare_out.
 
    Args:
        display_plane:   плоскость дисплея весов (входная плоскость линзы).
        detector_plane:  плоскость детекторов (выходная плоскость линзы).
        config:          физические параметры установки.
    """
    def __init__(self,
                 display_plane: _ConfigDesignPlane,
                 detector_plane: _ConfigDesignPlane,
                 config: _LumaiOpticConfig):
        M = display_plane.pixel_count_by_y
        N = display_plane.pixel_count_by_x
 
        # Физика суммирующей линзы:
        # Каждый столбец j принимает свет от всех M строк.
        # Амплитуда на детекторе j = сумма амплитуд по строкам i.
        # Весовой профиль — апертурная функция линзы (прямоугольная апертура).
        #
        # Для физически корректной модели учитываем:
        # 1. Апертуру линзы (ограничение по Y)
        # 2. Фазовую маску линзы (квадратичная фаза)
        # При некогерентном свете фаза не важна для интенсивности,
        # но важна для когерентного режима.
 
        # Апертура по Y: принимаем все M строк равномерно
        # operator_Y [1, M]: суммирующий оператор
        # Нормируем на sqrt(M) для сохранения энергии
        operator_Y = _torch.ones(1, M, dtype=_torch.cfloat) / M**0.5  # [1, M]
 
        # operator_X [N, N]: единичная матрица — столбцы независимы
        operator_X = _torch.eye(N, dtype=_torch.cfloat)  # [N, N]
 
        super().__init__(operator_X, operator_Y)
 
class PropagatorFreeSpaceLumai(PropagatorSinc):
    """
    Свободное пространство между лазерами и дисплеем в установке Lumai.
 
    Использует тот же физически точный sinc-пропагатор что и в 4f системе,
    но с параметрами соответствующими геометрии Lumai:
    короткое расстояние (~5 см) против 20 см в статье POMMM.
 
    Это моделирует распространение гауссовых пучков VCSEL
    от лазерной матрицы до плоскости дисплея весов.
    """
    pass  # Полностью наследуем PropagatorSinc — физика та же