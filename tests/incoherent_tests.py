"""Проверки некогерентной схемы (source/incoherent.py). Запуск: python -m tests.incoherent_tests"""
import warnings
import torch
from source import (IncoherentConfig, IncoherentOpticalMul, OpticalEngine, led_spectrum, GLASS,
                    IncoherentDevice, DeviceConfig, ModulatorConfig, DetectorConfig)


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


def test_glass_and_spectrum():
    assert abs(GLASS['N-BK7'](587.6e-9) - 1.5168) < 1e-4
    sp = led_spectrum(525e-9, 35e-9, 5)
    assert abs(sum(w for _, w in sp) - 1) < 1e-9 and len(sp) == 5


def test_mask_fill_pair_equals_scalar():
    X, W = torch.rand(1, 1, 3, 8), torch.rand(1, 1, 8, 6)
    a, b = small(mask_fill=0.75)(X, W), small(mask_fill=(0.75, 0.75))(X, W)
    assert ((a - b).norm() / a.norm()) < 1e-6


def test_lens_chromatism():
    """Одиночные линзы и спектр 35 нм размывают сильнее, чем ахромат (NA 0,1, элемент 50 мкм)."""
    X, W = torch.rand(1, 1, 5, 8), torch.rand(1, 1, 8, 6)
    sp = led_spectrum(525e-9, 35e-9, 5)
    achro = rel(small(wavelength=525e-9, spectrum=sp)(X, W), X @ W)
    single = rel(small(wavelength=525e-9, spectrum=sp, lens_glass='N-BK7')(X, W), X @ W)
    assert single > 2 * achro


def dev(m, **kw):
    return IncoherentDevice(m, DeviceConfig(**kw))


def test_ideal_device_is_identity():
    m = small()
    X, W = torch.rand(1, 1, 4, 8), torch.rand(1, 1, 8, 6)
    assert ((dev(m)(X, W) - m(X, W)).norm() / m(X, W).norm()) < 1e-6


def test_modulators():
    m = small()
    X, W = torch.rand(1, 1, 5, 8), torch.rand(1, 1, 8, 6)
    ideal = m(X, W)
    lcd = ModulatorConfig(levels=256, gamma=2.2, contrast=300)
    y = dev(m, source=lcd, mask=lcd)(X, W)
    assert rel(y, ideal) < 0.01                       # 8 бит и вычтенная утечка: < 1 %
    y = dev(m, source=lcd, mask=lcd, subtract_leak=False)(X, W)
    assert rel(y, ideal) > rel(dev(m, source=lcd, mask=lcd)(X, W), ideal)


def test_detector_noise_scales():
    m = small()
    X, W = torch.rand(1, 1, 64, 8), torch.rand(1, 1, 8, 6)
    ideal = m(X, W)
    errs = []
    for fw in (1e4, 1e6):
        torch.manual_seed(0)
        errs.append(rel(dev(m, detector=DetectorConfig(full_well=fw))(X, W), ideal))
    assert 5 < errs[0] / errs[1] < 20                 # дробовой шум ∝ 1/√N: ожидаем ~10


def test_device_gradients():
    m = small()
    lcd = ModulatorConfig(levels=256, gamma=2.2, contrast=300, nonuni=0.02)
    d = dev(m, source=lcd, mask=lcd, detector=DetectorConfig(full_well=1e5, read_noise=5, adc_bits=12))
    X = torch.rand(1, 1, 3, 8, requires_grad=True)
    W = torch.rand(1, 1, 8, 6, requires_grad=True)
    d(X, W).sum().backward()
    assert torch.isfinite(X.grad).all() and torch.isfinite(W.grad).all() and W.grad.abs().sum() > 0


def test_device_cached_response():
    m = small()
    lcd = ModulatorConfig(levels=256, gamma=2.2, contrast=300)
    d = dev(m, source=lcd, mask=lcd)
    X, W = torch.rand(1, 1, 3, 4), torch.rand(1, 1, 4, 3)
    assert ((d(X, W, response=d.response(W)) - d(X, W)).norm() / d(X, W).norm()) < 1e-6


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
