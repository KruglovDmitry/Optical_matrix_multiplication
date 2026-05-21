#include "field.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро: |E|²  → IntensityMap
//  Каждый поток обрабатывает один пиксель
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_intensity(const cuComplex* __restrict__ field,
                             float*           __restrict__ intensity,
                             int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    intensity[idx] = cabs2(field[idx]);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро: нормализация [0, max] → [0, 1]   (inplace, float)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void k_normalize(float* __restrict__ data, float inv_max, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] *= inv_max;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро: colormap "inferno" (аппроксимация, 4-й порядок)
//  input:  float  [0, 1]
//  output: uchar4 RGBA
// ─────────────────────────────────────────────────────────────────────────────
__device__ float3 inferno_colormap(float t)
{
    // Polynomial approximation of matplotlib "inferno"
    // Coefficients from: https://www.shadertoy.com/view/WlfXRN
    const float3 c0 = {0.0002189403691192265f,  0.001651004010331247f,  -0.01948089843709184f};
    const float3 c1 = {0.1065134194856116f,      0.5639564367884091f,     3.932712388889277f};
    const float3 c2 = {11.60249308247187f,       -3.972853965665698f,    -15.9423855558633f};
    const float3 c3 = {-41.70399613139459f,       17.43639888205313f,     44.35414519872813f};
    const float3 c4 = {77.162935699427f,         -33.40235894210092f,    -81.80730925738993f};
    const float3 c5 = {-70.839493217592f,         30.18814395214634f,     73.20951985803202f};
    const float3 c6 = {24.78428813521409f,        -9.832732866436185f,   -23.59598716042023f};

    return make_float3(
        c0.x + t*(c1.x + t*(c2.x + t*(c3.x + t*(c4.x + t*(c5.x + t*c6.x))))),
        c0.y + t*(c1.y + t*(c2.y + t*(c3.y + t*(c4.y + t*(c5.y + t*c6.y))))),
        c0.z + t*(c1.z + t*(c2.z + t*(c3.z + t*(c4.z + t*(c5.z + t*c6.z)))))
    );
}

__global__ void k_to_rgba(const float*  __restrict__ intensity,
                            unsigned char* __restrict__ rgba,
                            int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float t = __saturatef(intensity[idx]);   // clamp [0,1]
    float3 c = inferno_colormap(t);

    rgba[4*idx + 0] = (unsigned char)(__saturatef(c.x) * 255.f);
    rgba[4*idx + 1] = (unsigned char)(__saturatef(c.y) * 255.f);
    rgba[4*idx + 2] = (unsigned char)(__saturatef(c.z) * 255.f);
    rgba[4*idx + 3] = 255;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host API
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <vector>
#include <cstdio>

// stb_image_write — header-only, включаем один раз здесь
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void compute_intensity(const ComplexField& field,
                        IntensityMap&       out,
                        cudaStream_t stream)
{
    int N = field.nx * field.ny;
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    k_intensity<<<blocks, threads, 0, stream>>>(field.data, out.data, N);
    CUDA_CHECK(cudaGetLastError());
}

void normalize_intensity(IntensityMap& map, cudaStream_t stream)
{
    int N = map.nx * map.ny;

    // Находим максимум на CPU (для прототипа достаточно)
    std::vector<float> host(N);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), map.data,
                               sizeof(float)*N,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float mx = *std::max_element(host.begin(), host.end());
    if (mx < 1e-30f) return;

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    k_normalize<<<blocks, threads, 0, stream>>>(map.data, 1.f / mx, N);
    CUDA_CHECK(cudaGetLastError());
}

// Сохранение IntensityMap в PNG с colormap inferno
// scale_factor — для апскейла маленьких плоскостей (SLM 6×256 → читаемый PNG)
void save_intensity_png(const IntensityMap& map,
                         const std::string&  path,
                         int scale_factor,
                         cudaStream_t stream)
{
    int N = map.nx * map.ny;

    // 1. Скачиваем нормализованные интенсивности
    std::vector<float> host(N);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), map.data,
                               sizeof(float)*N,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 2. Конвертируем в RGBA на CPU (достаточно для сохранения)
    int W = map.nx * scale_factor;
    int H = map.ny * scale_factor;
    std::vector<unsigned char> img(W * H * 4);

    for (int j = 0; j < H; ++j) {
        for (int i = 0; i < W; ++i) {
            int src_i = i / scale_factor;
            int src_j = j / scale_factor;
            float t   = host[src_j * map.nx + src_i];
            t = std::max(0.f, std::min(1.f, t));

            // Inline inferno на CPU
            auto poly = [](float x, float c0, float c1, float c2,
                           float c3, float c4, float c5, float c6) {
                return c0 + x*(c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + x*c6)))));
            };

            unsigned char r = (unsigned char)(std::max(0.f, std::min(1.f,
                poly(t, 0.0002f, 0.1065f, 11.60f, -41.70f, 77.16f, -70.84f, 24.78f))) * 255);
            unsigned char g = (unsigned char)(std::max(0.f, std::min(1.f,
                poly(t, 0.0017f, 0.5640f, -3.973f, 17.44f, -33.40f, 30.19f, -9.83f))) * 255);
            unsigned char b = (unsigned char)(std::max(0.f, std::min(1.f,
                poly(t, -0.0195f, 3.933f, -15.94f, 44.35f, -81.81f, 73.21f, -23.60f))) * 255);

            int dst = (j * W + i) * 4;
            img[dst+0] = r;
            img[dst+1] = g;
            img[dst+2] = b;
            img[dst+3] = 255;
        }
    }

    int ok = stbi_write_png(path.c_str(), W, H, 4, img.data(), W * 4);
    if (!ok) {
        fprintf(stderr, "[intensity] Failed to write PNG: %s\n", path.c_str());
    } else {
        printf("[intensity] Saved %s  (%d x %d)\n", path.c_str(), W, H);
    }
}
