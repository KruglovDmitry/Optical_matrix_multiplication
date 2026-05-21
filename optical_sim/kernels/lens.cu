#include "propagator.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Скрещенная линза (PropagatorCrossLens)
//
//  По X: диагональная матрица diag(exp(-i K/(f) * x²))
//        точнее: exp(-i K / distance * x²)   [полная фаза, не /2]
//  По Y: диагональная матрица diag(exp(-i K / (2*distance) * y²))
//
//  В Python:
//    operator_X = exp(-i K / distance * x²)       (скрещенная по X: /f без /2)
//    operator_Y = exp(-i K / (2*distance) * y²)   (скрещенная по Y: /2f)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_cross_lens_x(cuComplex* __restrict__ op,
                                 int n, float dx, float ap,
                                 float K, float dist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = -ap * 0.5f + dx * (i + 0.5f);
    float phase = -K / dist * x * x;

    float cs, sn;
    sincosf(phase, &sn, &cs);

    // Диагональная матрица [n × n]: op[i][j] = delta_{ij} * exp(i*phase)
    // Храним как плотную матрицу (для единообразия с sinc-оператором)
    for (int j = 0; j < n; ++j) {
        op[i * n + j] = (i == j) ? make_cuComplex(cs, sn) : make_cuComplex(0.f, 0.f);
    }
}

__global__ void k_cross_lens_y(cuComplex* __restrict__ op,
                                 int n, float dy, float ap,
                                 float K, float dist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float y = -ap * 0.5f + dy * (i + 0.5f);
    float phase = -K / (2.f * dist) * y * y;

    float cs, sn;
    sincosf(phase, &sn, &cs);

    for (int j = 0; j < n; ++j) {
        op[i * n + j] = (i == j) ? make_cuComplex(cs, sn) : make_cuComplex(0.f, 0.f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Цилиндрическая линза (PropagatorCylindLens)
//
//  По X: то же, что cross_lens_x
//  По Y: единичная матрица (ones → identity)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_identity(cuComplex* __restrict__ op, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < n; ++j) {
        op[i * n + j] = (i == j) ? make_cuComplex(1.f, 0.f) : make_cuComplex(0.f, 0.f);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host API
// ─────────────────────────────────────────────────────────────────────────────

void build_cross_lens_x(Operator& op,
                          const Plane& plane,
                          const SystemConfig& cfg,
                          cudaStream_t stream)
{
    int n = plane.nx;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    k_cross_lens_x<<<blocks, threads, 0, stream>>>(
        op.data, n, plane.dx, plane.aperture_x(), cfg.K(), cfg.distance);
    CUDA_CHECK(cudaGetLastError());
}

void build_cross_lens_y(Operator& op,
                          const Plane& plane,
                          const SystemConfig& cfg,
                          cudaStream_t stream)
{
    int n = plane.ny;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    k_cross_lens_y<<<blocks, threads, 0, stream>>>(
        op.data, n, plane.dy, plane.aperture_y(), cfg.K(), cfg.distance);
    CUDA_CHECK(cudaGetLastError());
}

void build_cylind_lens_x(Operator& op,
                           const Plane& plane,
                           const SystemConfig& cfg,
                           cudaStream_t stream)
{
    // По X: то же самое, что скрещенная линза
    build_cross_lens_x(op, plane, cfg, stream);
}

void build_cylind_lens_y(Operator& op,
                           const Plane& plane,
                           const SystemConfig& cfg,
                           cudaStream_t stream)
{
    // По Y: единичная матрица
    int n = plane.ny;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    k_identity<<<blocks, threads, 0, stream>>>(op.data, n);
    CUDA_CHECK(cudaGetLastError());
}
