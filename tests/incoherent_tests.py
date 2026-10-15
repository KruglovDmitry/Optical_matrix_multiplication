"""Проверки некогерентной схемы (source/incoherent.py). Запуск: python -m tests.incoherent_tests"""
import warnings
import torch
from source import IncoherentConfig, IncoherentOpticalMul, OpticalEngine


def small(split=32, **kw):
    cfg = dict(n_in=8, n_out=6, pitch_x=50e-6, split_x=split, split_y=split,
               src_height=300e-6, distance=0.1, lens_pixel_size=2e-6, lens_size=10000)
    cfg.update(kw)
    return IncoherentOpticalMul(IncoherentConfig(**cfg))


def rel(y, ref):
    g = (y * ref).sum() / (y * y).sum()
    return ((g * y - ref).norm() / ref.norm()).item()


def test_orientation():
    """Светодиод j через ячейку (j, k) попадает на детектор k."""
    m = small()
    e = torch.eye(8)[None, None]
    W = torch.zeros(1, 1, 8, 6); W[0, 0, torch.arange(6), torch.arange(6)] = 1
    assert m(e, W)[0, 0].argmax(1)[:6].tolist() == list(range(6))


def test_linear_in_input():
    m = small()
    X1, X2, W = torch.rand(1, 1, 4, 8), torch.rand(1, 1, 4, 8), torch.rand(1, 1, 8, 6)
    a, b = m(X1 + 2 * X2, W), m(X1, W) + 2 * m(X2, W)
    assert ((a - b).norm() / b.norm()) < 1e-5


def test_accuracy_large_element():
    """u = p·NA/λ ≈ 9,4: положительное умножение точнее 1 %."""
    m = small()
    X, W = torch.rand(1, 1, 5, 8), torch.rand(1, 1, 8, 6)
    assert rel(m(X, W), X @ W) < 0.01


def test_smaller_matrix_centered():
    m = small()
    X, W = torch.rand(1, 1, 3, 4), torch.rand(1, 1, 4, 2)
    assert m(X, W).shape == (1, 1, 3, 2) and rel(m(X, W), X @ W) < 0.02


def test_crop_equals_full_field():
    """Расчёт на срезе совпадает с расчётом на всём поле с выключенными краями."""
    m = small()
    X, W = torch.rand(1, 1, 3, 5), torch.rand(1, 1, 5, 3)
    Xf = torch.nn.functional.pad(X, (1, 2))
    Wf = torch.nn.functional.pad(W, (1, 2, 1, 2))
    full = m(Xf, Wf)[..., 1:4]
    assert ((m(X, W) - full).norm() / full.norm()) < 1e-5


def test_grid_warning():
    m = small(split=8)          # δ = 6,25 мкм > λ/(8NA) = 0,66 мкм
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        assert not m.check_grid()['ok'] and w


def test_source_spread_covers_field():
    # Подточка светодиода 150 мкм светит только на ±0,18 мм, поле ±0,4 мм: края тёмные
    m = small(split=4, n_out=16, src_samples_y=2)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        assert not m.check_grid(warn=False)['ok']
    assert m.illumination_profile()['rows'][0] < 0.1
    n = IncoherentOpticalMul.min_src_samples(m.config)
    rows = small(split=4, n_out=16, src_samples_y=n).illumination_profile()['rows']
    assert (rows - 1).abs().max() < 0.05


def test_engine_split():
    cfg = IncoherentConfig(n_in=8, n_out=8, pitch_x=50e-6, split_x=32, split_y=32,
                           src_height=300e-6, distance=0.1, lens_pixel_size=2e-6, lens_size=10000)
    eng = OpticalEngine(size=8, physics='incoherent', incoherent_config=cfg)
    A, B = torch.randn(5, 8), torch.randn(8, 8)
    assert rel(eng.matmul(A, B), A @ B) < 0.05


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
