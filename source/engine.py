"""
optics.py — переиспользуемая обёртка над оптическим симулятором
для встраивания оптического матричного умножения в трансформер.

Ключевая идея — переключаемость: каждый матмул можно независимо гонять либо
через оптический sim, либо через обычный torch. Это позволяет одним и тем же
кодом собирать любую конфигурацию (всё оптическое / только внимание / только
линейные / полностью цифровой baseline) и честно их сравнивать.

Компоненты:
  * optics_matmul   — примитив (сдвиг → нормировка → sim → масштаб → поправки),
                      проверен численно против torch-матмула до ~1e-13;
  * OpticalEngine   — единый «движок»: один omm.OpticalMul, единый gain,
                      единый конфиг апертуры, автоматический тайлинг матриц > апертуры;
  * mm(...)         — диспетчер: optical → engine.matmul, иначе → torch.matmul;
  * OpticLinear     — линейный слой (вес [in, out]) с флагом optical;
  * OpticalAttention— весь блок внимания в обёртке; независимые флаги proj_optical
                      (q/k/v/o) и attn_optical (q@kᵀ и attn@v).

Контракт симулятора (README): вход — два 4D-тензора
    left (B, C, H, W) и right (B, C, W, K)  →  выход (B, C, H, K),
т.е. батч-матмул по двум ведущим осям. Движок приводит операнды к 4D сам.
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
SIM_GAIN_INV = 1.0 / 3.44e-3


def optics_matmul(sim, A, B, eps=1e-8, gain=1.0):
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
    """
    k = A.shape[-1]
    sa = torch.clamp(-A.amin(dim=-1, keepdim=True), min=0)   # [...,M,1]
    sb = torch.clamp(-B.amin(dim=-2, keepdim=True), min=0)   # [...,1,N]
    P, Q = A + sa, B + sb
    a = P.amax(dim=-1, keepdim=True).clamp_min(eps)
    b = Q.amax(dim=-2, keepdim=True).clamp_min(eps)
    PQ = sim(P / a, Q / b) * a * b * gain
    corr_a = sb * A.sum(dim=-1, keepdim=True)
    corr_b = sa * B.sum(dim=-2, keepdim=True)
    corr_c = k * sa * sb
    return PQ - corr_a - corr_b - corr_c


class OpticalEngine(nn.Module):
    """
    Единый источник правды для оптических матмулов: один omm.OpticalMul, конфиг
    апертуры и калибровка gain. Создаётся один раз на модель и передаётся во все слои.

    matmul(a, b) считает a @ b по последним двум осям:
      * a:[...,M,K], b:[...,K,N] (или b — 2D-вес [K,N], broadcast по батчу);
      * все размеры <= size → один optics_matmul;
      * иначе тайлинг: по M/N — блоки <= size и конкатенация; по контракции K —
        чанки и суммирование (матмул линеен по K, поправки по-чанково точны).
        Тайлинг физически честен: реальная апертура фиксирована.
    """
    def __init__(self, size=512, pixel_size=3.6e-6, gain_inv=SIM_GAIN_INV,
                 splits=2, distance=0.01):
        super().__init__()
        self.size = size
        self.gain = gain_inv
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

    def _tiled_matmul(self, a4, b4):
        s = self.size
        M, K = a4.shape[-2], a4.shape[-1]
        N = b4.shape[-1]
        rows = []
        for i in range(0, M, s):
            cols = []
            for j in range(0, N, s):
                acc = None
                for kk in range(0, K, s):
                    p = optics_matmul(self.sim,
                                      a4[..., i:i + s, kk:kk + s],
                                      b4[..., kk:kk + s, j:j + s],
                                      gain=self.gain)
                    acc = p if acc is None else acc + p
                cols.append(acc)
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)

    def matmul(self, a, b):
        a4, b4 = self._to_4d(a), self._to_4d(b)
        if max(a4.shape[-2], a4.shape[-1], b4.shape[-1]) <= self.size:
            out4 = optics_matmul(self.sim, a4, b4, gain=self.gain)
        else:
            out4 = self._tiled_matmul(a4, b4)
        return out4[0] if a.dim() == 3 else out4


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
            tril = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(tril == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = mm(self.engine, attn, v, self.attn_optical)
        return self.o_proj(self._gather(out, B, T))