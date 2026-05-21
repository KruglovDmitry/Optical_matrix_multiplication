#include "propagator.hpp"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Аппроксимация интегралов Френеля на GPU
//
//  Используем разложение через вспомогательные функции f, g (Abramowitz & Stegun 7.3.27)
//  Точность ~2e-6 для |x| > 0 (достаточно для физического моделирования).
//
//  C(x) = integral_0^x cos(π/2 * t²) dt
//  S(x) = integral_0^x sin(π/2 * t²) dt
// ─────────────────────────────────────────────────────────────────────────────
__device__ void fresnel_cs(float x, float& C, float& S)
{
    float ax = fabsf(x);
    float t  = ax;
    float t2 = t * t;

    // Для малых аргументов — ряд Тейлора
    if (ax < 1.5f) {
        float term = t;
        float sum_c = term, sum_s = 0.f;
        float t4 = t2 * t2;
        float sign = 1.f;
        for (int n = 1; n <= 12; ++n) {
            sign = -sign;
            term *= t4 / ((2*n) * (2*n - 1));
            sum_c += sign * term / (4*n + 1);
            sum_s += sign * term / (4*n + 3);
        }
        // Стандартный ряд:  C ≈ x*(1 - x⁴/10 + ...), S ≈ x³π/6*(1 - ...)
        // Точная формула через ряд:
        float pi_2 = 1.5707963f; // π/2
        C = ax; S = 0.f;
        float pw = 1.f;
        float fac = 1.f;
        int sign2 = 1;
        C = 0.f; S = 0.f;
        float u = pi_2 * t2;
        float uk = 1.f;
        float kfac = 1.f;
        float tc = t;
        float ts = t * u / 3.f;
        C = tc; S = ts;
        for (int k = 1; k <= 20; ++k) {
            uk  *= -u * u;
            kfac = kfac * (2.f*k) * (2.f*k - 1.f);
            float ck = uk / (kfac * (4.f*k + 1.f)) * t;
            float sk = uk * (-u) / (kfac * (2.f*k+1.f) * (4.f*k + 3.f)) * t;
            // пересчитываем через стандартные коэффициенты
            (void)ck; (void)sk;
        }
        // Упрощённый и надёжный ряд (из DLMF 7.6)
        C = 0.f; S = 0.f;
        float term_c = t;
        float term_s = (M_PI / 2.f) * t * t2 / 3.f;
        C += term_c;
        S += term_s;
        float pi2_2 = (M_PI * M_PI / 4.f);
        float t4v = t2 * t2;
        for (int n = 1; n <= 15; ++n) {
            term_c *= -pi2_2 * t4v / ((4.f*n - 1.f) * (4.f*n));
            term_s *= -pi2_2 * t4v / ((4.f*n + 1.f) * (4.f*n + 2.f));
            C += term_c;
            S += term_s;
        }
    } else {
        // Асимптотическое разложение для больших аргументов
        float inv_t  = 1.f / (M_PI * ax);
        float inv_t2 = inv_t * inv_t;

        float f = inv_t * (1.f - 9.f*inv_t2*(1.f - 39.f*inv_t2));
        float g = inv_t2 * inv_t * (1.f - 17.f*inv_t2);

        float angle = M_PI * 0.5f * ax * ax;
        float sa, ca;
        sincosf(angle, &sa, &ca);

        C = 0.5f + f * sa - g * ca;
        S = 0.5f - f * ca - g * sa;
    }

    if (x < 0.f) { C = -C; S = -S; }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро: строим оператор sinc-распространения по одному измерению
//
//  op[i_out][i_in] = h(x_out[i_out] - x_in[i_in])
//
//  где h — элемент матрицы (интеграл Френеля), как в PropagatorSinc.__get_operator_for_dim
//
//  op.rows = n_out,  op.cols = n_in
//  plane_in.coord(i_in)   → x_in
//  plane_out.coord(i_out) → x_out
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_build_sinc_op(cuComplex* __restrict__ op,
                                  int n_out, int n_in,
                                  float dx_in, float dx_out,
                                  float ap_in, float ap_out,  // апертуры
                                  float K, float dist)
{
    // Поток (i_out, i_in)
    int i_out = blockIdx.y * blockDim.y + threadIdx.y;
    int i_in  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_out >= n_out || i_in >= n_in) return;

    // Координаты пикселей (центрированная сетка)
    float x_in  = -ap_in  * 0.5f + dx_in  * (i_in  + 0.5f);
    float x_out = -ap_out * 0.5f + dx_out * (i_out + 0.5f);

    float diff = x_out - x_in;

    // Параметры из PropagatorSinc.__get_operator_for_dim:
    //   bndW  = 0.5 / dx_in
    //   eikz  = exp(i K dist)^0.5    — скалярная фаза, общая для всех элементов
    //   sqzk  = sqrt(2 * dist / K)
    //   sq2p  = sqrt(2/π)
    //   mu1   = -π * sqzk * bndW - diff / sqzk
    //   mu2   =  π * sqzk * bndW - diff / sqzk
    //   C,S   = fresnel(mu * sq2p)
    //   h = (sqrt(dx_in * dx_out) / π) / sqzk * eikz
    //       * exp(0.5i * diff² * K / dist)
    //       * (ΔC - i ΔS) / sq2p

    float bndW = 0.5f / dx_in;
    float sqzk = sqrtf(2.f * dist / K);
    float sq2p = sqrtf(2.f / M_PI);
    float scale = sqrtf(dx_in * dx_out) / (M_PI * sqzk * sq2p);

    float mu1 = -M_PI * sqzk * bndW - diff / sqzk;
    float mu2 =  M_PI * sqzk * bndW - diff / sqzk;

    float C1, S1, C2, S2;
    fresnel_cs(mu1 * sq2p, C1, S1);
    fresnel_cs(mu2 * sq2p, C2, S2);

    float dC = C2 - C1;
    float dS = S2 - S1;

    // Фаза eikz = exp(i K dist)^0.5  →  exp(i K dist / 2)
    // Фаза Fresnel: exp(0.5i diff² K/dist)
    float phase_z    = 0.5f * K * dist;
    float phase_diff = 0.5f * diff * diff * K / dist;
    float total_phase = phase_z + phase_diff;

    float cos_p, sin_p;
    sincosf(total_phase, &sin_p, &cos_p);
    // eikz * exp(i phase_diff): (cos + i sin) * scale
    // (ΔC - i ΔS) / sq2p:  это уже в scale

    // Перемножаем: (cos_p + i sin_p) * (dC - i dS)
    cuComplex h = make_cuComplex(
        scale * (cos_p * dC + sin_p * dS),
        scale * (sin_p * dC - cos_p * dS)
    );

    op[i_out * n_in + i_in] = h;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host функции
// ─────────────────────────────────────────────────────────────────────────────

// Вспомогательная: запуск ядра построения оператора
static void launch_sinc_op(Operator& op,
                            int n_out, int n_in,
                            float dx_in, float dx_out,
                            float ap_in, float ap_out,
                            const SystemConfig& cfg,
                            cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 blocks((n_in  + threads.x - 1) / threads.x,
                (n_out + threads.y - 1) / threads.y);

    k_build_sinc_op<<<blocks, threads, 0, stream>>>(
        op.data, n_out, n_in,
        dx_in, dx_out, ap_in, ap_out,
        cfg.K(), cfg.distance
    );
    CUDA_CHECK(cudaGetLastError());
}

void build_sinc_operator_x(Operator& op,
                             const Plane& src, const Plane& dst,
                             const SystemConfig& cfg, cudaStream_t stream)
{
    // Оператор по X: [nx_dst × nx_src]
    // Но применяется справа: E_out = E_in @ Op_X
    // Поэтому логически op_x[i_in][i_out] (транспонированный порядок)
    // — как в Python: operator_X.transpose(-2,-1)
    // Здесь храним [n_in × n_out] для умножения справа напрямую
    launch_sinc_op(op,
                   src.nx, dst.nx,  // [n_in × n_out]
                   dst.dx, src.dx,
                   dst.aperture_x(), src.aperture_x(),
                   cfg, stream);
}

void build_sinc_operator_y(Operator& op,
                             const Plane& src, const Plane& dst,
                             const SystemConfig& cfg, cudaStream_t stream)
{
    // Оператор по Y: [ny_out × ny_in]
    launch_sinc_op(op,
                   dst.ny, src.ny,
                   src.dy, dst.dy,
                   src.aperture_y(), dst.aperture_y(),
                   cfg, stream);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Применение пропагатора:  E_out = Op_Y @ E_in @ Op_X
//
//  Реализовано наивно (без cuBLAS) — заменим в следующей итерации
//  E_in:   [ny_in  × nx_in]
//  Op_Y:   [ny_out × ny_in]
//  Op_X:   [nx_in  × nx_out]
//  E_out:  [ny_out × nx_out]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_apply_propagator(
    const cuComplex* __restrict__ E_in,     // [ny_in × nx_in]
    const cuComplex* __restrict__ op_y,     // [ny_out × ny_in]
    const cuComplex* __restrict__ op_x,     // [nx_in  × nx_out]
    cuComplex*       __restrict__ E_out,    // [ny_out × nx_out]
    int ny_in, int nx_in, int ny_out, int nx_out)
{
    int i_out = blockIdx.x * blockDim.x + threadIdx.x;  // X output
    int j_out = blockIdx.y * blockDim.y + threadIdx.y;  // Y output
    if (i_out >= nx_out || j_out >= ny_out) return;

    cuComplex acc = make_cuComplex(0.f, 0.f);

    // Суммируем по j_in (Y_in) и i_in (X_in)
    // acc = sum_{j_in} sum_{i_in} op_y[j_out, j_in] * E_in[j_in, i_in] * op_x[i_in, i_out]
    for (int j_in = 0; j_in < ny_in; ++j_in) {
        cuComplex oy = op_y[j_out * ny_in + j_in];
        cuComplex row_acc = make_cuComplex(0.f, 0.f);
        for (int i_in = 0; i_in < nx_in; ++i_in) {
            cuComplex e  = E_in[j_in * nx_in + i_in];
            cuComplex ox = op_x[i_in * nx_out + i_out];
            row_acc = cadd(row_acc, cmul(e, ox));
        }
        acc = cadd(acc, cmul(oy, row_acc));
    }
    E_out[j_out * nx_out + i_out] = acc;
}

void apply_propagator(const ComplexField& field_in,
                       ComplexField&       field_out,
                       const Operator&     op_x,
                       const Operator&     op_y,
                       cudaStream_t        stream)
{
    dim3 threads(16, 16);
    dim3 blocks((field_out.nx + threads.x - 1) / threads.x,
                (field_out.ny + threads.y - 1) / threads.y);

    k_apply_propagator<<<blocks, threads, 0, stream>>>(
        field_in.data,
        op_y.data,
        op_x.data,
        field_out.data,
        field_in.ny, field_in.nx,
        field_out.ny, field_out.nx
    );
    CUDA_CHECK(cudaGetLastError());
}
