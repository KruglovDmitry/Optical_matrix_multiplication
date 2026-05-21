#pragma once
#include "field.hpp"
#include "config.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Оператор распространения (матрица на GPU)
//
//  В Python-версии оператор хранится как матрица [N×M] cuComplex.
//  Применение:  E_out = Op_Y @ E_in @ Op_X
//  Op_Y:  [ny_out × ny_in]  — распространение по Y
//  Op_X:  [nx_in  × nx_out] — распространение по X (транспонированный порядок!)
// ─────────────────────────────────────────────────────────────────────────────
struct Operator {
    cuComplex* data = nullptr;
    int rows = 0, cols = 0;   // логические размеры матрицы

    static Operator allocate(int rows_, int cols_) {
        Operator op;
        op.rows = rows_; op.cols = cols_;
        CUDA_CHECK(cudaMalloc(&op.data, sizeof(cuComplex) * rows_ * cols_));
        CUDA_CHECK(cudaMemset(op.data, 0, sizeof(cuComplex) * rows_ * cols_));
        return op;
    }

    void free() {
        if (data) { cudaFree(data); data = nullptr; }
    }

    // Элемент [r][c]
    __device__ cuComplex& at(int r, int c) { return data[r * cols + c]; }
    __device__ const cuComplex& at(int r, int c) const { return data[r * cols + c]; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Объявления CUDA-ядер для построения операторов
//  (реализация в kernels/fresnel.cu и kernels/lens.cu)
// ─────────────────────────────────────────────────────────────────────────────

// Оператор Синк-распространения в свободном пространстве
// Эквивалент PropagatorSinc из Python
void build_sinc_operator_x(Operator& op,
                            const Plane& src,
                            const Plane& dst,
                            const SystemConfig& cfg,
                            cudaStream_t stream = 0);

void build_sinc_operator_y(Operator& op,
                            const Plane& src,
                            const Plane& dst,
                            const SystemConfig& cfg,
                            cudaStream_t stream = 0);

// Оператор скрещенной линзы (PropagatorCrossLens)
// Диагональная матрица: op = diag(exp(-i K/f * x²))
void build_cross_lens_x(Operator& op,
                         const Plane& plane,
                         const SystemConfig& cfg,
                         cudaStream_t stream = 0);

void build_cross_lens_y(Operator& op,
                         const Plane& plane,
                         const SystemConfig& cfg,
                         cudaStream_t stream = 0);

// Оператор цилиндрической линзы (PropagatorCylindLens)
// По X: то же, что cross_lens; по Y: единичная матрица
void build_cylind_lens_x(Operator& op,
                          const Plane& plane,
                          const SystemConfig& cfg,
                          cudaStream_t stream = 0);

void build_cylind_lens_y(Operator& op,
                          const Plane& plane,
                          const SystemConfig& cfg,
                          cudaStream_t stream = 0);

// ─────────────────────────────────────────────────────────────────────────────
//  Применение оператора:  E_out = Op_Y @ E_in @ Op_X
//  E_in:  [ny_in  × nx_in]
//  Op_Y:  [ny_out × ny_in]   → умножение слева
//  Op_X:  [nx_in  × nx_out]  → умножение справа
//  E_out: [ny_out × nx_out]
// ─────────────────────────────────────────────────────────────────────────────
void apply_propagator(const ComplexField& field_in,
                       ComplexField&       field_out,
                       const Operator&     op_x,
                       const Operator&     op_y,
                       cudaStream_t stream = 0);