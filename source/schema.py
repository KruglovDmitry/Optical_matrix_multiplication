"""
schema.py — отрисовка физической схемы установки из объектов конфигурации.
Ничего не выдумывает: все дистанции, апертуры и число элементов берутся из
Config (геометрия тракта) и VCSELArraySpec (линейка/приёмный тракт), поэтому
картинка всегда синхронна с тем, что реально считает сим.

Запуск: python schema.py  -> scheme.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
from pathlib import Path

try:
    from .config import Config
    from .partial_coherence import VCSELArraySpec
except ImportError:
    import sys
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[1]))
    from source.config import Config
    from source.partial_coherence import VCSELArraySpec

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

def draw_scheme(config: Config, spec: VCSELArraySpec,
                out_path: str = "scheme.png") -> str:
    d = config.distance
    planes = [
        ("VCSEL-\nмассив", config.input_vector_plane, 0.0, "#d97706"),
        ("Скрещ.\nлинза 1", config.first_lens_plane, d, "#2563eb"),
        ("SLM\n(матрица W)", config.matrix_plane, 2 * d, "#059669"),
        ("Скрещ.\nлинза 2", config.second_lens_plane, 3 * d, "#2563eb"),
        ("PD-массив\n+ АЦП", config.output_vector_plane, 4 * d, "#dc2626"),
    ]
    max_h = max(p[1].aperture_height for p in planes)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for name, plane, z, color in planes:
        h = plane.aperture_height / max_h * 1.6 + 0.25
        ax.add_patch(Rectangle((z * 1000 - 0.4, -h / 2), 0.8, h,
                               facecolor=color, alpha=0.75, edgecolor="k",
                               linewidth=0.6, zorder=3))
        ax.text(z * 1000, h / 2 + 0.18, name, ha="center", va="bottom",
                fontsize=9)
        ax.text(z * 1000, -h / 2 - 0.18,
                f"{plane.pixel_count_by_y}×{plane.pixel_count_by_x} px\n"
                f"{plane.aperture_height*1e3:.2f}×"
                f"{plane.aperture_width*1e3:.2f} мм",
                ha="center", va="top", fontsize=7, color="#444")
    for i in range(len(planes) - 1):
        z0, z1 = planes[i][2] * 1000, planes[i + 1][2] * 1000
        ax.add_patch(FancyArrow(z0 + 0.6, 0, z1 - z0 - 1.5, 0, width=0.008,
                                head_width=0.09, head_length=0.5,
                                color="#999", zorder=1))
        ax.text((z0 + z1) / 2, 0.12, f"{d*1000:.0f} мм", ha="center",
                fontsize=7, color="#666")

    info = (f"λ = {config.wavelength*1e9:.0f} нм   "
            f"N = {spec.n_emitters} излучателей   "
            f"Δν_min = {spec.dnu_min/1e9:.1f} ГГц   "
            f"f_mod = {spec.f_mod/1e9:.1f} ГГц   "
            f"f_АЦП = {spec.f_adc/1e6:.0f} МГц   "
            f"T_int = {spec.t_int*1e9:.1f} нс")
    ax.set_title("Некогерентная схема MVM (из Config + VCSELArraySpec)\n"
                 + info, fontsize=9)
    ax.set_xlabel("z, мм")
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlim(-4, 4 * d * 1000 + 4)
    ax.get_yaxis().set_visible(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


cfg = make_config(wavelength=850e-9)
spec = VCSELArraySpec(n_emitters=16)
print(draw_scheme(cfg, spec))