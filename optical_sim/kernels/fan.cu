#include "field.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Fan-out ядро
//
//  Цилиндрическая линза размножает каждый источник по всем строкам SLM.
//  Вход: field_in  [1 × nx]         — одна строка (один источник)
//  Выход: field_out [num_rows × nx]  — одна и та же строка, повторённая num_rows раз
//
//  В реальной системе это реализуется физически цилиндрической линзой fan-out.
//  В симуляторе: просто broadcast по строкам.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_fan_out(const cuComplex* __restrict__ src,   // [1 × nx]
                            cuComplex*       __restrict__ dst,   // [num_rows × nx]
                            int nx, int num_rows)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;  // пиксель по X
    int row = blockIdx.y;                               // строка SLM
    if (i >= nx || row >= num_rows) return;

    dst[row * nx + i] = src[i];
}

// ─────────────────────────────────────────────────────────────────────────────
//  Fan-in ядро
//
//  Суммирует вклады всех источников на каждом детекторе:
//  PD_j = sum_i |E_out[j, i_range]|²
//
//  Вход: field_in [num_sources × nx_det]
//  Выход: intensity [num_detectors]
//
//  Каждый детектор соответствует одному источнику (диагональный случай).
//  Для общего случая используется матрица соответствия.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_fan_in(const cuComplex* __restrict__ field,   // [num_sources × nx]
                           float*           __restrict__ detectors, // [num_detectors]
                           int nx, int num_sources, int num_detectors)
{
    int det = blockIdx.x * blockDim.x + threadIdx.x;
    if (det >= num_detectors) return;

    // Каждый детектор суммирует интенсивность по всей строке
    // (соответствующей источнику с тем же индексом)
    int src_row = det % num_sources;

    float sum = 0.f;
    for (int i = 0; i < nx; ++i) {
        sum += cabs2(field[src_row * nx + i]);
    }
    detectors[det] = sum;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро: умножение поля на SLM маску (поэлементное)
//  field [ny × nx], mask [ny × nx] → field *= mask  (inplace)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_apply_slm(cuComplex*       __restrict__ field,  // [ny × nx]
                              const cuComplex* __restrict__ mask,   // [ny × nx]
                              int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    field[idx] = cmul(field[idx], mask[idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host API
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdio>

void fan_out(const ComplexField& src,
              ComplexField&       dst,
              cudaStream_t        stream)
{
    // src: [1 × nx],  dst: [num_rows × nx]
    dim3 threads(256, 1);
    dim3 blocks((src.nx + threads.x - 1) / threads.x, dst.ny);
    k_fan_out<<<blocks, threads, 0, stream>>>(
        src.data, dst.data, src.nx, dst.ny);
    CUDA_CHECK(cudaGetLastError());
}

void apply_slm(ComplexField&       field,
                const ComplexField& mask,
                cudaStream_t        stream)
{
    int N = field.nx * field.ny;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    k_apply_slm<<<blocks, threads, 0, stream>>>(
        field.data, mask.data, N);
    CUDA_CHECK(cudaGetLastError());
}

void fan_in(const ComplexField& field,
             float*              detectors_gpu,
             int                 num_detectors,
             cudaStream_t        stream)
{
    int threads = 256;
    int blocks  = (num_detectors + threads - 1) / threads;
    k_fan_in<<<blocks, threads, 0, stream>>>(
        field.data, detectors_gpu,
        field.nx, field.ny, num_detectors);
    CUDA_CHECK(cudaGetLastError());
}
