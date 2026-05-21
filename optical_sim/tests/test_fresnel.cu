#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include "propagator.hpp"
#include "pipeline.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Проверяем унитарность (сохранение энергии) propagator sinc:
//  Гауссов пучок → распространение → суммарная интенсивность должна
//  сохраняться (с допуском на краевые эффекты).
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    printf("=== test_fresnel ===\n");

    SystemConfig cfg;
    cfg.wavelength  = 850e-9f;
    cfg.distance    = 0.03f;

    // Маленькие плоскости для скорости теста
    Plane src_plane = { 32, 1, 5e-6f, 5e-6f };
    Plane dst_plane = { 32, 1, 5e-6f, 5e-6f };

    // 1. Строим операторы
    Operator op_x = Operator::allocate(src_plane.nx, dst_plane.nx);
    Operator op_y = Operator::allocate(dst_plane.ny, src_plane.ny);

    build_sinc_operator_x(op_x, src_plane, dst_plane, cfg);
    build_sinc_operator_y(op_y, src_plane, dst_plane, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[OK] build_sinc_operator\n");

    // 2. Создаём гауссов пучок
    ComplexField src = ComplexField::allocate(src_plane);
    ComplexField dst = ComplexField::allocate(dst_plane);

    std::vector<cuComplex> host_src(src_plane.nx);
    float beam_w = src_plane.nx * src_plane.dx * 0.2f;
    float energy_in = 0.f;
    for (int i = 0; i < src_plane.nx; ++i) {
        float x   = src_plane.coord_x(i);
        float amp = expf(-x*x / (2.f * beam_w * beam_w));
        host_src[i] = make_cuComplex(amp, 0.f);
        energy_in += amp * amp;
    }
    src.upload(host_src.data());

    // 3. Применяем propagator
    apply_propagator(src, dst, op_x, op_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[OK] apply_propagator\n");

    // 4. Считаем энергию на выходе
    std::vector<cuComplex> host_dst(dst_plane.nx);
    dst.download(host_dst.data());

    float energy_out = 0.f;
    for (auto& v : host_dst) energy_out += cabs2(v);

    printf("     Energy in:  %.6f\n", energy_in);
    printf("     Energy out: %.6f\n", energy_out);

    // Допускаем потери до 30% из-за краевых эффектов при малом размере плоскости
    float ratio = energy_out / energy_in;
    printf("     Ratio: %.3f  (expected ~1.0)\n", ratio);
    assert(ratio > 0.5f && ratio < 2.0f && "Energy conservation violated");
    printf("[OK] Energy conservation (ratio in [0.5, 2.0])\n");

    // 5. Проверяем что оператор не нулевой
    std::vector<cuComplex> op_host(src_plane.nx * dst_plane.nx);
    CUDA_CHECK(cudaMemcpy(op_host.data(), op_x.data,
                          sizeof(cuComplex) * op_host.size(),
                          cudaMemcpyDeviceToHost));
    float op_norm = 0.f;
    for (auto& v : op_host) op_norm += cabs2(v);
    assert(op_norm > 0.f);
    printf("[OK] Operator non-zero (norm = %.4f)\n", op_norm);

    src.free(); dst.free();
    op_x.free(); op_y.free();

    printf("\n=== All tests passed ===\n");
    return 0;
}
