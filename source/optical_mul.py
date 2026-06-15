import torch as _torch
import torch.nn as _nn
from .config import Config as _Config
from .propagator import PropagatorCrossLens as _PropCrossLens, PropagatorCylindLens as _PropСylindLens, PropagatorSinc as _PropSinc, Propagator as _Prop
from .propagator import (
    PropagatorTrainableCylindLens as _PropagatorTrainableCylindLens,
    PropagatorTrainableFocalDistCylindLens as _PropagatorTrainableFocalDistCylindLens
)
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import matplotlib.pyplot as plt

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
    
class TrainableLensOpticalMul(_nn.Module):
    """
    Класс системы, выполняющей оптически операцию умножения матрицы на матрицу.
    """
    def __init__(self, config: _Config):
        """
        Конструктор класса.
 
        Args:
            config: конфигурация расчётной системы.
        """
        super().__init__()

        prop_one = _PropSinc(config.input_vector_plane, config.first_lens_plane, config)
        prop_two = _PropCrossLens(config.first_lens_plane, config)
        prop_three = _PropSinc(config.first_lens_plane, config.matrix_plane, config)
        prop_four = _PropagatorTrainableCylindLens(config.matrix_plane, config)
        prop_five = _PropSinc(config.matrix_plane, config.second_lens_plane, config)
        prop_six = _PropCrossLens(config.second_lens_plane, config).T
        prop_seven = _PropSinc(config.second_lens_plane, config.output_vector_plane, config)

        self._propagator_one: _Prop = prop_one + prop_two + prop_three 
        self._propagator_cylind_lens: _Prop = prop_four
        self._propagator_three: _Prop = prop_five + prop_six + prop_seven

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
        # TODO data should be at least two seq length. For one we get
        # untimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
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
        vec_field = self._propagator_cylind_lens(vec_field, mat_field.shape[-2:])
        vec_field = self._propagator_three(vec_field * mat_field, (mat_field.size(-2), 1))

        return self.prepare_out(vec_field)

    @_torch.no_grad()
    def log_cylind_lens_operator_x(
        self,
        writer: SummaryWriter,
        tag: str,
        global_step: Optional[int] = None,
    ):
        # 1. Apply exp to get the wrapped phase as it would be physically seen
        # This ensures values outside [-pi, pi] wrap correctly
        complex_op = _torch.exp(-1j * self._propagator_cylind_lens._operator_X_phi)
        wrapped_phase = _torch.angle(complex_op).float() # Range: [-π, π]
        
        # 2. Normalize for Image Visualization [0, 1]
        phase_normalized = (wrapped_phase + _torch.pi) / (2 * _torch.pi)
        
        # 3. Log as a 1-pixel high row
        # Shape: [1, 1, Width]
        phase_row = phase_normalized.unsqueeze(0).unsqueeze(0)
        writer.add_image(f"{tag}/phase_row", phase_row, global_step, dataformats='CHW')

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wrapped_phase.detach().cpu().numpy(), label=f'Step {global_step}')
        ax.set_title(f"Cylindrical Lens Phase Profile (x) {tag}")
        ax.set_xlabel("Pixel Index")
        ax.set_ylabel("Phase (rad)")
        ax.set_ylim([-_torch.pi - 0.5, _torch.pi + 0.5])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        
        # Send the figure to the "Plots" or "Images" tab in TensorBoard
        writer.add_figure(f"{tag}/phase_profile", fig, global_step)
        plt.close(fig) # Important: prevent memory leaks


class TrainableFocalDistLensOpticalMul(_nn.Module):
    """
    Класс системы, выполняющей оптически операцию умножения матрицы на матрицу.
    """
    def __init__(self, config: _Config):
        """
        Конструктор класса.
 
        Args:
            config: конфигурация расчётной системы.
        """
        super().__init__()

        prop_one = _PropSinc(config.input_vector_plane, config.first_lens_plane, config)
        prop_two = _PropCrossLens(config.first_lens_plane, config)
        prop_three = _PropSinc(config.first_lens_plane, config.matrix_plane, config)
        prop_four = _PropagatorTrainableFocalDistCylindLens(config.matrix_plane, config)
        prop_five = _PropSinc(config.matrix_plane, config.second_lens_plane, config)
        prop_six = _PropCrossLens(config.second_lens_plane, config).T
        prop_seven = _PropSinc(config.second_lens_plane, config.output_vector_plane, config)

        self._propagator_one: _Prop = prop_one + prop_two + prop_three 
        self._propagator_cylind_lens: _Prop = prop_four
        self._propagator_three: _Prop = prop_five + prop_six + prop_seven

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
        # TODO data should be at least two seq length. For one we get
        # untimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
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
        vec_field = self._propagator_cylind_lens(vec_field, mat_field.shape[-2:])
        vec_field = self._propagator_three(vec_field * mat_field, (mat_field.size(-2), 1))

        return self.prepare_out(vec_field)

    @_torch.no_grad()
    def log_cylind_lens_operator_x(
        self,
        writer: SummaryWriter,
        tag: str,
        global_step: Optional[int] = None,
    ):
        # 1. Apply exp to get the wrapped phase as it would be physically seen
        # This ensures values outside [-pi, pi] wrap correctly
        lens = self._propagator_cylind_lens
        writer.add_scalar(f"{tag}/focal_distance", lens._distance.detach().cpu().numpy(), global_step)

        complex_op = _torch.exp(-1j * lens._K / lens._distance * lens._linspace_by_x**2)
        wrapped_phase = _torch.angle(complex_op).float() # Range: [-π, π]
        
        # 2. Normalize for Image Visualization [0, 1]
        phase_normalized = (wrapped_phase + _torch.pi) / (2 * _torch.pi)
        
        # 3. Log as a 1-pixel high row
        # Shape: [1, 1, Width]
        phase_row = phase_normalized.unsqueeze(0).unsqueeze(0)
        writer.add_image(f"{tag}/phase_row", phase_row, global_step, dataformats='CHW')

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(wrapped_phase.detach().cpu().numpy(), label=f'Step {global_step}')
        ax.set_title(f"Cylindrical Lens Phase Profile (x) {tag}\nFocal distance: {lens._distance.detach().cpu().numpy()}")
        ax.set_xlabel("Pixel Index")
        ax.set_ylabel("Phase (rad)")
        ax.set_ylim([-_torch.pi - 0.5, _torch.pi + 0.5])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Send the figure to the "Plots" or "Images" tab in TensorBoard
        writer.add_figure(f"{tag}/phase_profile", fig, global_step)
        plt.close(fig) # Important: prevent memory leaks