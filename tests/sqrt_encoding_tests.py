"""
T7: проверка рецепта оптиков «подавать корни значений».
Кодировка: SLM = sqrt(W) (амплитудное пропускание), вход = интенсивности x.
Ожидание: y_detector ~ c * (x @ W) с точностью до дифракционного кросс-тока
ВНУТРИ каждого излучателя (свет одного источника на соседних пикселях SLM
взаимно когерентен и интерферирует — геометрическая картинка оптиков это
опускает). Меряем: относительный остаток после подгонки скаляра c, и его
зависимость от split-факторов (качество дискретизации поля).
"""
import torch
torch.manual_seed(0)

from pc_engine import OpticalMul, TransferMatrix, PartialCoherentMul
from test_partial_coherence import make_config, PIX
from source.config import Config


def run_case(n=16, splits=2, distance=0.01, trials=8):
    cfg = Config(right_matrix_count_columns=n, right_matrix_count_rows=n,
                 right_matrix_width=PIX * n, right_matrix_height=PIX * n,
                 min_height_gap=PIX,
                 right_matrix_split_x=splits, right_matrix_split_y=splits,
                 left_matrix_split_x=splits, left_matrix_split_y=splits,
                 result_matrix_split=splits, distance=distance,
                 wavelength=850e-9)
    mul = OpticalMul(cfg)
    errs = []
    for _ in range(trials):
        W = torch.rand(n, n)                       # значения матрицы, [0,1]
        x = torch.rand(n)                          # значения вектора, [0,1]
        tm = TransferMatrix(mul, W.sqrt())         # <-- рецепт оптиков
        pcm = PartialCoherentMul(tm)
        y = pcm.incoherent(x).flatten()            # интенсивностное чтение
        y_ref = x @ W                              # целевой матмул в значениях
        c = (y @ y_ref) / (y_ref @ y_ref)          # подгонка калибр. скаляра
        errs.append(((y / c - y_ref).norm() / y_ref.norm()).item())
    e = torch.tensor(errs)
    return e.mean().item(), e.std().item()


if __name__ == "__main__":
    print("T7: sqrt-кодировка, остаток к x @ W после калибровки скаляра")
    for splits in (1, 2, 4):
        m, s = run_case(splits=splits)
        print(f"  split={splits}:  eps = {m:.4f} ± {s:.4f}")
    # контроль: та же метрика для КОГЕРЕНТНОЙ схемы (амплитудная кодировка),
    # чтобы сравнить уровни остаточной ошибки двух архитектур на одной сетке
    cfg = make_config(wavelength=850e-9)
    mul = OpticalMul(cfg)
    errs = []
    for _ in range(8):
        W = torch.rand(16, 16); x = torch.rand(16)
        y = mul(x.view(1, 1, 1, 16), W.view(1, 1, 16, 16)).flatten()
        y_ref = x @ W
        c = (y @ y_ref) / (y_ref @ y_ref)
        errs.append(((y / c - y_ref).norm() / y_ref.norm()).item())
    e = torch.tensor(errs)
    print(f"  когерентная схема (контроль): eps = {e.mean():.4f} ± {e.std():.4f}")