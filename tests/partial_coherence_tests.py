"""
Исполняемая спецификация ветки partial-coherence.
Запуск: python test_partial_coherence.py  (из каталога с source/ или рядом с repo/)

Тесты:
  T1. T воспроизводит когерентный forward репо (бит-в-бит по пайплайну).
  T2. Некогерентный режим == честный поэлементный прогон (сумма интенсивностей).
  T3. Пределы gamma: gamma=1 -> когерентная интенсивность, gamma=0 -> некогерентная.
  T4. Закон усреднения eps ~ 1/sqrt(M).
  T5. Биения: при dnu >= f_adc + 2*f_mod остаточная ошибка мала, при dnu ~ 0 — велика.
  T6. Чувствительность |T|^2 к lambda (спред линейки 6 пм и 1 нм).
"""
import math
import sys
import time

import torch

torch.manual_seed(0)

from pc_engine import (Config, OpticalMul, PartialCoherentMul, TransferMatrix,
                       VCSELArraySpec, crosstalk_error, epsilon_vs_averaging,
                       wavelength_sensitivity)

PIX = 3.6e-6


def make_config(n_rows=16, n_cols=16, wavelength=532e-9):
    return Config(right_matrix_count_columns=n_cols,
                  right_matrix_count_rows=n_rows,
                  right_matrix_width=PIX * n_cols,
                  right_matrix_height=PIX * n_rows,
                  min_height_gap=PIX,
                  right_matrix_split_x=2, right_matrix_split_y=2,
                  left_matrix_split_x=2, left_matrix_split_y=2,
                  result_matrix_split=2,
                  wavelength=wavelength,
                  distance=0.01)


def main():
    t0 = time.time()
    cfg = make_config()
    mul = OpticalMul(cfg)
    N = 16
    Wmat = torch.rand(N, N)                      # веса на SLM, [0,1]
    x = torch.rand(N)                            # входной вектор

    tm = TransferMatrix(mul, Wmat)
    pcm = PartialCoherentMul(tm)
    ok = True

    # --- T1: когерентный кросс-чек ------------------------------------------
    y_repo = mul(x.view(1, 1, 1, N), Wmat.view(1, 1, N, N)).flatten()
    y_T = pcm.coherent(x).flatten()
    e1 = crosstalk_error(y_T, y_repo)
    ok &= e1 < 1e-4
    print(f"T1 coherent vs repo forward:      eps = {e1:.2e}  "
          f"{'PASS' if e1 < 1e-4 else 'FAIL'}")

    # --- T2: некогерентный == поэлементная сумма интенсивностей -------------
    x_int = x                                     # интенсивности излучателей
    i_sum = None
    for j in range(N):
        e_j = torch.zeros(N); e_j[j] = math.sqrt(x_int[j])   # амплитуда
        f = torch.einsum('oi,i->o', tm.T, e_j.cfloat())
        i_j = f.abs().square()
        i_sum = i_j if i_sum is None else i_sum + i_j
    y_loop = tm.read_intensity(i_sum).flatten()
    y_inc = pcm.incoherent(x_int).flatten()
    e2 = crosstalk_error(y_inc, y_loop)
    ok &= e2 < 1e-5
    print(f"T2 incoherent vs element-wise:    eps = {e2:.2e}  "
          f"{'PASS' if e2 < 1e-5 else 'FAIL'}")

    # --- T3: пределы gamma ---------------------------------------------------
    g1 = torch.ones(N, N)
    y_g1 = pcm.partial(x_int, g1).flatten()
    f_coh = torch.einsum('oi,i->o', tm.T, x_int.sqrt().cfloat())
    y_coh_int = tm.read_intensity(f_coh.abs().square()).flatten()
    e3a = crosstalk_error(y_g1, y_coh_int)
    g0 = torch.eye(N)
    y_g0 = pcm.partial(x_int, g0).flatten()
    e3b = crosstalk_error(y_g0, y_inc)
    ok &= e3a < 1e-4 and e3b < 1e-5
    print(f"T3 gamma=1 -> coherent:           eps = {e3a:.2e}  "
          f"{'PASS' if e3a < 1e-4 else 'FAIL'}")
    print(f"   gamma=0 -> incoherent:         eps = {e3b:.2e}  "
          f"{'PASS' if e3b < 1e-5 else 'FAIL'}")

    # --- T4: закон 1/sqrt(M) -------------------------------------------------
    m_list = [1, 4, 16, 64]
    eps = epsilon_vs_averaging(pcm, x_int, gamma_scalar=1.0, m_list=m_list,
                               n_repeats=8)
    lx = [math.log(m) for m in m_list]
    ly = [math.log(e) for e in eps]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    slope = (sum((a - mx) * (b - my) for a, b in zip(lx, ly))
             / sum((a - mx) ** 2 for a in lx))
    law_ok = -0.65 < slope < -0.35                # ожидаем ~ -0.5
    ok &= law_ok
    print(f"T4 eps(M) for M={m_list}: "
          + ", ".join(f"{e:.3f}" for e in eps)
          + f"  slope = {slope:.2f} (ожид. -0.5)  "
          f"{'PASS' if law_ok else 'FAIL'}")

    # --- T5: биения и спектральный спейсинг ---------------------------------
    spec_good = VCSELArraySpec(n_emitters=N, dnu_min=2.5e9, dnu_sigma=0.0,
                               linewidth=100e6, f_mod=1e9, f_adc=500e6,
                               detector_bw=500e6)
    spec_bad = VCSELArraySpec(n_emitters=N, dnu_min=0.0, dnu_sigma=50e6,
                              linewidth=100e6, f_mod=1e9, f_adc=500e6,
                              detector_bw=500e6, seed=1)
    y_good = pcm.beats(x_int, spec_good, n_windows=4).flatten()
    y_bad = pcm.beats(x_int, spec_bad, n_windows=4).flatten()
    e5a = crosstalk_error(y_good, y_inc)
    e5b = crosstalk_error(y_bad, y_inc)
    ok &= e5a < 0.05 and e5b > 3 * e5a
    print(f"T5 beats, dnu=2.5GHz spacing:     eps = {e5a:.3f}  "
          f"{'PASS' if e5a < 0.05 else 'FAIL'}")
    print(f"   beats, dnu ~ 50MHz (in-band):  eps = {e5b:.3f}  "
          f"{'PASS' if e5b > 3 * e5a else 'FAIL'} (ожидаемо плохо)")
    print(f"   spec rule dnu_min >= f_adc+2f_mod = "
          f"{spec_good.required_dnu_min()/1e9:.1f} GHz")

    # --- T6: чувствительность к lambda ---------------------------------------
    lam0 = 850e-9
    sens = wavelength_sensitivity(
        lambda lam: make_config(wavelength=lam), Wmat,
        [lam0, lam0 + 6e-12, lam0 + 1e-9])
    print(f"T6 |T|^2 shift: d_lambda=6pm -> {sens[1]:.2e}, "
          f"1nm -> {sens[2]:.2e}  (вход в допуск спреда линейки)")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}   "
          f"({time.time() - t0:.1f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())