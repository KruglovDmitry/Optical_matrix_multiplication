#include "propagator.hpp"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Интегралы Френеля на GPU (DLMF 7.6)
//  C(x) = integral_0^x cos(π/2 * t²) dt
//  S(x) = integral_0^x sin(π/2 * t²) dt
// ─────────────────────────────────────────────────────────────────────────────
__device__ void fresnel_cs(float x, float& C, float& S)
{
    float ax = fabsf(x);
    float t2 = ax * ax;
    float pi2_2 = (M_PI * M_PI / 4.f);
    float t4 = t2 * t2;

    if (ax < 1.5f) {
        // Ряд Тейлора (DLMF 7.6.2)
        float term_c = ax;
        float term_s = (M_PI / 2.f) * ax * t2 / 3.f;
        C = term_c;
        S = term_s;
        for (int n = 1; n <= 15; ++n) {
            term_c *= -pi2_2 * t4 / ((4.f*n - 1.f) * (4.f*n));
            term_s *= -pi2_2 * t4 / ((4.f*n + 1.f) * (4.f*n + 2.f));
            C += term_c;
            S += term_s;
        }
    } else {
        // Асимптотическое разложение (DLMF 7.12.2)
        float inv_t  = 1.f / (M_PI * ax);
        float inv_t2 = inv_t * inv_t;
        float f = inv_t  * (1.f - 9.f * inv_t2);
        float g = inv_t2 * inv_t * (1.f - 17.f * inv_t2);
        float angle = M_PI * 0.5f * ax * ax;
        float sa, ca;
        sincosf(angle, &sa, &ca);
        C = 0.5f + f * sa - g * ca;
        S = 0.5f - f * ca - g * sa;
    }

    if (x < 0.f) { C = -C; S = -S; }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро: строим оператор sinc-распространения
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_build_sinc_op(cuComplex* __restrict__ op,
                                  int n_out, int n_in,
                                  float dx_in, float dx_out,
                                  float ap_in, float ap_out,
                                  float K, float dist)
{
    int i_out = blockIdx.y * blockDim.y + threadIdx.y;
    int i_in  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_out >= n_out || i_in >= n_in) return;

    float x_in  = -ap_in  * 0.5f + dx_in  * (i_in  + 0.5f);
    float x_out = -ap_out * 0.5f + dx_out * (i_out + 0.5f);
    float diff  = x_out - x_in;

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

    float phase = 0.5f * K * dist + 0.5f * diff * diff * K / dist;
    float cos_p, sin_p;
    sincosf(phase, &sin_p, &cos_p);

    op[i_in * n_out + i_out] = make_cuComplex(
        scale * (cos_p * dC + sin_p * dS),
        scale * (sin_p * dC - cos_p * dS)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host функции
// ─────────────────────────────────────────────────────────────────────────────
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
        cfg.K(), cfg.distance);
    CUDA_CHECK(cudaGetLastError());
}

void build_sinc_operator_x(Operator& op,
                             const Plane& src, const Plane& dst,
                             const SystemConfig& cfg, cudaStream_t stream)
{
    launch_sinc_op(op, dst.nx, src.nx,
                   src.dx, dst.dx,
                   src.aperture_x(), dst.aperture_x(),
                   cfg, stream);
}

void build_sinc_operator_y(Operator& op,
                             const Plane& src, const Plane& dst,
                             const SystemConfig& cfg, cudaStream_t stream)
{
    // Для плоскостей с ny=1 оператор по Y — единичная матрица
    if (src.ny == 1 && dst.ny == 1) {
        cuComplex one = make_cuComplex(1.f, 0.f);
        CUDA_CHECK(cudaMemcpy(op.data, &one, sizeof(cuComplex),
                              cudaMemcpyHostToDevice));
        return;
    }
    launch_sinc_op(op, dst.ny, src.ny,
                   src.dy, dst.dy,
                   src.aperture_y(), dst.aperture_y(),
                   cfg, stream);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Применение пропагатора: E_out = Op_Y @ E_in @ Op_X
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_apply_propagator(
    const cuComplex* __restrict__ E_in,
    const cuComplex* __restrict__ op_y,
    const cuComplex* __restrict__ op_x,
    cuComplex*       __restrict__ E_out,
    int ny_in, int nx_in, int ny_out, int nx_out)
{
    int i_out = blockIdx.x * blockDim.x + threadIdx.x;
    int j_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_out >= nx_out || j_out >= ny_out) return;

    cuComplex acc = make_cuComplex(0.f, 0.f);
    for (int j_in = 0; j_in < ny_in; ++j_in) {
        cuComplex oy = op_y[j_out * ny_in + j_in];
        cuComplex row_acc = make_cuComplex(0.f, 0.f);
        for (int i_in = 0; i_in < nx_in; ++i_in) {
            row_acc = cadd(row_acc, cmul(E_in[j_in * nx_in + i_in],
                                         op_x[i_in * nx_out + i_out]));
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
        field_in.data, op_y.data, op_x.data, field_out.data,
        field_in.ny, field_in.nx, field_out.ny, field_out.nx);
    CUDA_CHECK(cudaGetLastError());
}