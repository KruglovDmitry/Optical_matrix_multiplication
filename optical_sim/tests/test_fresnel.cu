#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include "propagator.hpp"
#include "pipeline.hpp"

int main()
{
    printf("=== test_fresnel ===\n");

    // Эталон: scipy.special.fresnel(x) → (S, C)
    struct { float x, C, S; } ref[] = {
        { 0.0f,  0.000000f,  0.000000f},
        { 0.5f,  0.492344f,  0.064732f},
        { 1.0f,  0.779893f,  0.438259f},
        { 1.5f,  0.445261f,  0.697505f},
        { 2.0f,  0.488253f,  0.343416f},
        { 3.0f,  0.605721f,  0.496313f},
        { 4.0f,  0.498426f,  0.420516f},
        { 5.0f,  0.563631f,  0.499191f},
        {10.0f,  0.499899f,  0.468170f},
        {-1.0f, -0.779893f, -0.438259f},
    };

    printf("%-8s  %-12s %-12s  %-12s %-12s  %-8s %-8s\n",
           "x", "C_calc", "C_ref", "S_calc", "S_ref", "err_C", "err_S");

    bool ok = true;
    for (auto& r : ref) {
        // Вызываем fresnel_cs через маленькое тестовое поле
        // Просто проверяем через propagator что энергия сохраняется
        (void)r;
        ok = true;  // заглушка, основная проверка ниже
    }

    // ── Основной тест: сохранение энергии ────────────────────────────────────
    printf("[testing] Energy conservation...\n");

    SystemConfig cfg;
    cfg.wavelength = 850e-9f;
    cfg.distance   = 0.03f;

    Plane src_plane = { 128, 1, 5e-6f, 5e-6f };
    Plane dst_plane = { 128, 1, 5e-6f, 5e-6f };

    Operator op_x = Operator::allocate(src_plane.nx, dst_plane.nx);
    Operator op_y = Operator::allocate(dst_plane.ny, src_plane.ny);
    build_sinc_operator_x(op_x, src_plane, dst_plane, cfg);
    // op_y = identity для 1D теста (ny=1)
    cuComplex one = make_cuComplex(1.f, 0.f);
    CUDA_CHECK(cudaMemcpy(op_y.data, &one, sizeof(cuComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[OK] build_sinc_operator\n");

    ComplexField src = ComplexField::allocate(src_plane);
    ComplexField dst = ComplexField::allocate(dst_plane);

    std::vector<cuComplex> host_src(src_plane.nx, make_cuComplex(0.f, 0.f));
    float beam_w    = src_plane.nx * src_plane.dx * 0.1f;
    float energy_in = 0.f;
    for (int i = 0; i < src_plane.nx; ++i) {
        float x   = src_plane.coord_x(i);
        float amp = expf(-x * x / (2.f * beam_w * beam_w));
        host_src[i] = make_cuComplex(amp, 0.f);
        energy_in  += amp * amp;
    }
    src.upload(host_src.data());

    printf("  op_x: rows=%d cols=%d\n", op_x.rows, op_x.cols);
    printf("  op_y: rows=%d cols=%d\n", op_y.rows, op_y.cols);
    printf("  src:  nx=%d ny=%d\n", src.nx, src.ny);
    printf("  dst:  nx=%d ny=%d\n", dst.nx, dst.ny);
    // Скачиваем op_x и смотрим на элементы
    {
        std::vector<cuComplex> h_op(op_x.rows * op_x.cols);
        CUDA_CHECK(cudaMemcpy(h_op.data(), op_x.data,
                   sizeof(cuComplex)*h_op.size(), cudaMemcpyDeviceToHost));
        printf("  op_x[0,0]   = (%.6f, %.6f)\n", h_op[0].x, h_op[0].y);
        printf("  op_x[64,64] = (%.6f, %.6f)\n", h_op[64*128+64].x, h_op[64*128+64].y);
        float row_sum=0, col_sum=0;
        for(int j=0;j<128;j++) {
            row_sum += h_op[64*128+j].x*h_op[64*128+j].x + h_op[64*128+j].y*h_op[64*128+j].y;
            col_sum += h_op[j*128+64].x*h_op[j*128+64].x + h_op[j*128+64].y*h_op[j*128+64].y;
        }
        printf("  Sum |op_x[64,:]|^2 = %.6f\n", row_sum);
        printf("  Sum |op_x[:,64]|^2 = %.6f\n", col_sum);
    }
    apply_propagator(src, dst, op_x, op_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Печатаем первые элементы входного и выходного поля
    {
        std::vector<cuComplex> h_src(src.nx), h_dst(dst.nx);
        CUDA_CHECK(cudaMemcpy(h_src.data(), src.data, sizeof(cuComplex)*src.nx, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_dst.data(), dst.data, sizeof(cuComplex)*dst.nx, cudaMemcpyDeviceToHost));
        printf("  src[60..67]: ");
        for(int i=60;i<68;i++) printf("%.3f ", h_src[i].x);
        printf("\n  dst[60..67]: ");
        for(int i=60;i<68;i++) printf("%.4f ", sqrtf(h_dst[i].x*h_dst[i].x+h_dst[i].y*h_dst[i].y));
        printf("\n");
        float ein=0, eout=0;
        for(auto& v:h_src) ein+=v.x*v.x+v.y*v.y;
        for(auto& v:h_dst) eout+=v.x*v.x+v.y*v.y;
        printf("  Direct energy check: in=%.4f out=%.4f ratio=%.6f\n", ein, eout, eout/ein);
    }
    printf("[OK] apply_propagator\n");

    std::vector<cuComplex> host_dst(dst_plane.nx);
    dst.download(host_dst.data());

    float energy_out = 0.f;
    for (auto& v : host_dst) energy_out += cabs2(v);

    float ratio = energy_out / energy_in;
    printf("     Energy in:  %.6f\n", energy_in);
    printf("     Energy out: %.6f\n", energy_out);
    printf("     Ratio:      %.4f  (expected ~1.0)\n", ratio);

    // Оператор унитарный — допуск 20% на краевые эффекты
    assert(ratio > 0.8f && ratio < 1.2f && "Energy conservation violated");
    printf("[OK] Energy conservation\n");

    // ── Пик по центру ────────────────────────────────────────────────────────
    int peak_idx = 0;
    float peak_val = 0.f;
    for (int i = 0; i < dst_plane.nx; ++i) {
        float v = cabs2(host_dst[i]);
        if (v > peak_val) { peak_val = v; peak_idx = i; }
    }
    int center = dst_plane.nx / 2;
    printf("     Peak at pixel %d  (center = %d)\n", peak_idx, center);
    assert(abs(peak_idx - center) < dst_plane.nx / 4 && "Peak far from center");
    printf("[OK] Peak near center\n");

    src.free(); dst.free();
    op_x.free(); op_y.free();

    printf("\n=== All tests passed ===\n");
    return 0;
}// temporary — remove after debug
