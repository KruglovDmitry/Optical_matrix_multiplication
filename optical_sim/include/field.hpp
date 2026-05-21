#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include "config.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Вспомогательные операции с cuComplex (inline, используются в ядрах и хосте)
// ─────────────────────────────────────────────────────────────────────────────
__host__ __device__ inline cuComplex make_polar(float amp, float phase) {
    return make_cuComplex(amp * cosf(phase), amp * sinf(phase));
}

__host__ __device__ inline float cabs2(cuComplex z) {
    return z.x * z.x + z.y * z.y;   // |z|²
}

__host__ __device__ inline cuComplex cmul(cuComplex a, cuComplex b) {
    return make_cuComplex(a.x * b.x - a.y * b.y,
                          a.x * b.y + a.y * b.x);
}

__host__ __device__ inline cuComplex cadd(cuComplex a, cuComplex b) {
    return make_cuComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline cuComplex cscale(cuComplex a, float s) {
    return make_cuComplex(a.x * s, a.y * s);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Макрос проверки CUDA ошибок
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error at ")             \
                + __FILE__ + ":" + std::to_string(__LINE__) + " — "            \
                + cudaGetErrorString(_e));                                      \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
//  ComplexField — световое поле на одной плоскости
//
//  Раскладка памяти: row-major [ny × nx]
//  data[j * nx + i]  =  поле в пикселе (i, j), i — X, j — Y
// ─────────────────────────────────────────────────────────────────────────────
struct ComplexField {
    cuComplex* data = nullptr;  // GPU память
    int nx = 0, ny = 0;        // размер плоскости
    Plane plane{};              // метаданные (шаг сетки и т.д.)

    // Число элементов
    __host__ int size() const { return nx * ny; }

    // Аллокация на GPU
    static ComplexField allocate(const Plane& p) {
        ComplexField f;
        f.nx    = p.nx;
        f.ny    = p.ny;
        f.plane = p;
        CUDA_CHECK(cudaMalloc(&f.data, sizeof(cuComplex) * p.nx * p.ny));
        CUDA_CHECK(cudaMemset(f.data, 0,  sizeof(cuComplex) * p.nx * p.ny));
        return f;
    }

    // Освобождение GPU памяти
    void free() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
    }

    // Копирование host → device (src — плоский массив cuComplex размером nx*ny)
    void upload(const cuComplex* src) {
        CUDA_CHECK(cudaMemcpy(data, src,
                              sizeof(cuComplex) * nx * ny,
                              cudaMemcpyHostToDevice));
    }

    // Копирование device → host
    void download(cuComplex* dst) const {
        CUDA_CHECK(cudaMemcpy(dst, data,
                              sizeof(cuComplex) * nx * ny,
                              cudaMemcpyDeviceToHost));
    }

    // Обнуление
    void zero() {
        CUDA_CHECK(cudaMemset(data, 0, sizeof(cuComplex) * nx * ny));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  IntensityMap — карта интенсивности |E|² на GPU (float, [ny × nx])
//  Используется для визуализации и записи PNG
// ─────────────────────────────────────────────────────────────────────────────
struct IntensityMap {
    float* data = nullptr;
    int nx = 0, ny = 0;

    static IntensityMap allocate(int nx_, int ny_) {
        IntensityMap m;
        m.nx = nx_; m.ny = ny_;
        CUDA_CHECK(cudaMalloc(&m.data, sizeof(float) * nx_ * ny_));
        CUDA_CHECK(cudaMemset(m.data, 0, sizeof(float) * nx_ * ny_));
        return m;
    }

    void free() {
        if (data) { cudaFree(data); data = nullptr; }
    }

    void download(float* dst) const {
        CUDA_CHECK(cudaMemcpy(dst, data,
                              sizeof(float) * nx * ny,
                              cudaMemcpyDeviceToHost));
    }
};