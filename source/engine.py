import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from source import OpticalMul, Config

SIM_GAIN_INV = 1.0 / 3.44e-3

def mm_shift(sim, A, B, gain=1.0, eps=1e-8):
    """Восстановление знака сдвигом. 1 вызов sim. A:[...,M,K], B:[...,K,N].

    Численно неустойчиво на центрированных данных: считает малый A@B как
    разность больших PQ и поправок (катастрофическое сокращение). Ошибка sim
    усиливается в kappa = |PQ|/|A@B| раз (десятки-сотни на знаковых данных).
    """
    k = A.shape[-1]
    sa = torch.clamp(-A.amin(dim=-1, keepdim=True), min=0)
    sb = torch.clamp(-B.amin(dim=-2, keepdim=True), min=0)
    P, Q = A + sa, B + sb
    a = P.amax(dim=-1, keepdim=True).clamp_min(eps)
    b = Q.amax(dim=-2, keepdim=True).clamp_min(eps)
    PQ = sim((P / a).contiguous(), (Q / b).contiguous()) * a * b * gain
    corr_a = sb * A.sum(dim=-1, keepdim=True)
    corr_b = sa * B.sum(dim=-2, keepdim=True)
    corr_c = k * sa * sb
    return PQ - corr_a - corr_b - corr_c


def mm_split(sim, A, B, gain=1.0):
    """Восстановление знака разделением на +/-. 4 вызова sim.

    Численно устойчиво: каждая часть неотрицательна без сдвига, нет большого
    общего сокращения. Аналог индустриального dual-rail кодирования.
    """
    A_pos = torch.clamp(A, min=0); A_neg = torch.clamp(-A, min=0)
    B_pos = torch.clamp(B, min=0); B_neg = torch.clamp(-B, min=0)
    out_shape = A.shape[:-1] + (B.shape[-1],)
    zero = A.new_zeros(out_shape)

    def term(X, Y):
        mx, my = torch.max(X), torch.max(Y)
        if mx > 0 and my > 0:
            return sim((X / mx).contiguous(), (Y / my).contiguous()) * mx * my * gain
        return zero
    return (term(A_pos, B_pos) - term(A_pos, B_neg)
            - term(A_neg, B_pos) + term(A_neg, B_neg))


FORMULAS = {'shift': mm_shift, 'split': mm_split}


class OpticalEngine(nn.Module):
    def __init__(self, size=512, pixel_size=3.6e-6, gain_inv=SIM_GAIN_INV,
                 splits=2, distance=0.15, lens_size=16384,
                 formula='split', tile_size=None):
        super().__init__()
        assert formula in FORMULAS, f"формула {formula} не из {list(FORMULAS)}"
        self.size = size
        self.tile_size = min(tile_size or size, size)
        self.gain = gain_inv
        self.formula_name = formula
        self.formula = FORMULAS[formula]
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
                   lens_size=lens_size))
        # --- опциональная диагностика ---
        self._diag = collections.defaultdict(lambda: collections.defaultdict(float))
        self._diag_on = False

    # ---- диагностика (опциональная) ----
    def diag_on(self):
        """Включить сбор ошибки и знаковости по именованным матмулам."""
        self._diag.clear(); self._diag_on = True

    def diag_off(self):
        self._diag_on = False

    def pop_diag(self):
        """Вернуть накопленную диагностику по тегам и очистить.
        Формат: {tag: {rel_err, neg_frac, A_min, A_max, B_min, B_max, n}}."""
        out = {}
        for tag, d in self._diag.items():
            n = max(d['n'], 1)
            out[tag] = dict(rel_err=d['rel_err']/n*100, neg_frac=d['neg_frac']/n,
                            A_min=d['A_min'], A_max=d['A_max'],
                            B_min=d['B_min'], B_max=d['B_max'], n=int(d['n']))
        self._diag.clear()
        return out

    def _record(self, tag, A, B, out):
        if not self._diag_on or tag is None:
            return
        with torch.no_grad():
            ref = torch.matmul(A, B)
            rel = ((out - ref).norm() / ref.norm().clamp_min(1e-9)).item()
            d = self._diag[tag]
            if d['n'] == 0:
                d['A_min'] = A.amin().item(); d['A_max'] = A.amax().item()
                d['B_min'] = B.amin().item(); d['B_max'] = B.amax().item()
            else:
                d['A_min'] = min(d['A_min'], A.amin().item())
                d['A_max'] = max(d['A_max'], A.amax().item())
                d['B_min'] = min(d['B_min'], B.amin().item())
                d['B_max'] = max(d['B_max'], B.amax().item())
            d['rel_err'] += rel; d['n'] += 1
            d['neg_frac'] += (((A < 0).float().mean().item()
                               + (B < 0).float().mean().item()) / 2)

    @staticmethod
    def _to_4d(t):
        if t.dim() == 2:
            return t[None, None]
        if t.dim() == 3:
            return t[None]
        return t

    def _call(self, a4, b4):
        """Один вызов выбранной формулы (тайл или целая матрица)."""
        return self.formula(self.sim, a4, b4, gain=self.gain)

    def _tiled(self, a4, b4):
        """Последовательный тайлинг: блоки по M/N, суммирование по K."""
        s = self.tile_size
        M, K = a4.shape[-2], a4.shape[-1]
        N = b4.shape[-1]
        rows = []
        for i in range(0, M, s):
            cols = []
            for j in range(0, N, s):
                acc = None
                for kk in range(0, K, s):
                    p = self._call(a4[..., i:i+s, kk:kk+s],
                                   b4[..., kk:kk+s, j:j+s])
                    acc = p if acc is None else acc + p
                cols.append(acc)
            rows.append(torch.cat(cols, dim=-1))
        return torch.cat(rows, dim=-2)

    def matmul(self, a, b, tag=None):
        in_dtype = a.dtype

        if in_dtype in (torch.bfloat16, torch.float16):
            a, b = a.float(), b.float()
        a4, b4 = self._to_4d(a), self._to_4d(b)

        if max(a4.shape[-2], a4.shape[-1], b4.shape[-1]) <= self.tile_size:
            out4 = self._call(a4, b4)
        else:
            out4 = self._tiled(a4, b4)

        if self._diag_on:
            self._record(tag, a4, b4, out4)

        if a.dim() == 2:
            out = out4[0, 0]
        elif a.dim() == 3:
            out = out4[0]
        else:
            out = out4
        return out.to(in_dtype) if out.dtype != in_dtype else out

    @torch.no_grad()
    def calibrate_gain(self, n_trials=8, device=None, dtype=torch.float32):
        """Автокалибровка скалярного gain по реперным матрицам полной апертуры.
        gain = <sim, ref> / <sim, sim> — переводит выход детектора в мат. единицы."""
        num = den = 0.0
        for _ in range(n_trials):
            P = torch.rand(1, 1, self.size, self.size, device=device, dtype=dtype)
            Q = torch.rand(1, 1, self.size, self.size, device=device, dtype=dtype)
            raw = self.sim(P, Q); ref = torch.matmul(P, Q)
            num += (raw * ref).sum().item(); den += (raw * raw).sum().item()
        self.gain = num / den
        return self.gain


class OpticLinear(nn.Module):
    def __init__(self, in_features, out_features, engine=None, optical=True,
                 bias=True, tag=None, device=None, dtype=None):
        super().__init__()
        assert not (optical and engine is None), "optical=True требует engine"
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.optical = optical
        self.tag = tag
        self._engine = (engine,)
        self.weight = nn.Parameter(torch.empty((in_features, out_features), **fk))
        self.bias = nn.Parameter(torch.empty(out_features, **fk)) if bias else None
        if bias is False:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @property
    def engine(self):
        return self._engine[0]

    def set_optical(self, flag):
        """Переключить слой оптика<->цифра (для инференс-экспериментов)."""
        assert not (flag and self.engine is None), "optical=True требует engine"
        self.optical = flag

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.optical:
            y = self.engine.matmul(x, self.weight, tag=self.tag)
        else:
            y = torch.matmul(x, self.weight)
        return y + self.bias if self.bias is not None else y

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, optical={self.optical}, tag={self.tag}")