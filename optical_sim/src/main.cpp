#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>

#include "config.hpp"
#include "field.hpp"
#include "propagator.hpp"
#include "pipeline.hpp"

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  Вспомогательная: сохраняем поле плоскости как PNG
// ─────────────────────────────────────────────────────────────────────────────
static void save_plane(const ComplexField& field,
                        const std::string&  name,
                        const SystemConfig& cfg,
                        int                 scale = 4)
{
    IntensityMap imap = IntensityMap::allocate(field.nx, field.ny);
    compute_intensity(field, imap);
    normalize_intensity(imap);
    std::string path = cfg.output_dir + "/" + name + ".png";
    save_intensity_png(imap, path, scale);
    imap.free();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Создание тестового поля источника (гауссов пучок по X, точечный по Y)
//
//  Каждый VCSEL — гауссово распределение амплитуды по X
//  с шириной beam_w. Источники расположены с шагом source_pitch по Y.
// ─────────────────────────────────────────────────────────────────────────────
static ComplexField make_source_field(const SystemConfig& cfg)
{
    ComplexField f = ComplexField::allocate(cfg.source_plane);

    std::vector<cuComplex> host(cfg.source_plane.nx * cfg.source_plane.ny,
                                make_cuComplex(0.f, 0.f));

    float beam_w = cfg.source_pitch * 0.4f;   // ширина пучка ≈ 40% шага

    for (int j = 0; j < cfg.source_plane.ny; ++j) {
        float y = cfg.source_plane.coord_y(j);

        // Ближайший источник к данной строке Y
        // (источники расположены с шагом source_pitch, центрированы)
        float y_src = (j - (cfg.num_sources - 1) * 0.5f) * cfg.source_pitch;
        float dy    = y - y_src;

        // Амплитуда в Y — гауссова от расстояния до центра источника
        float amp_y = expf(-dy * dy / (2.f * beam_w * beam_w));

        for (int i = 0; i < cfg.source_plane.nx; ++i) {
            float x   = cfg.source_plane.coord_x(i);
            float amp_x = expf(-x * x / (2.f * beam_w * beam_w));
            host[j * cfg.source_plane.nx + i] = make_cuComplex(amp_x * amp_y, 0.f);
        }
    }

    f.upload(host.data());
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Создание SLM маски (случайная фаза для теста)
// ─────────────────────────────────────────────────────────────────────────────
static ComplexField make_slm_mask(const SystemConfig& cfg)
{
    Plane sp = cfg.slm_plane();
    ComplexField mask = ComplexField::allocate(sp);

    std::vector<cuComplex> host(sp.nx * sp.ny);
    srand(42);
    for (auto& v : host) {
        float phase = ((float)rand() / RAND_MAX) * 2.f * M_PI;
        v = make_cuComplex(cosf(phase), sinf(phase));
    }
    mask.upload(host.data());
    return mask;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Вывод информации об устройстве
// ─────────────────────────────────────────────────────────────────────────────
static void print_device_info()
{
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("┌─────────────────────────────────────────┐\n");
    printf("│  Device: %-32s│\n", prop.name);
    printf("│  SM count:  %-3d  │  Compute: %d.%d         │\n",
           prop.multiProcessorCount, prop.major, prop.minor);
    printf("│  VRAM: %5.1f GB                          │\n",
           prop.totalGlobalMem / 1e9);
    printf("└─────────────────────────────────────────┘\n\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    printf("=== Optical Simulator v0.1 ===\n\n");
    print_device_info();

    // --- Конфигурация ---
    SystemConfig cfg;
    if (argc > 1) cfg.num_sources = std::atoi(argv[1]);
    if (argc > 2) cfg.slm_cols   = std::atoi(argv[2]);
    cfg.slm_rows = cfg.num_sources;   // строк SLM = число источников

    printf("Sources: %d   SLM: %d × %d\n\n",
           cfg.num_sources, cfg.slm_rows, cfg.slm_cols);

    // --- Создаём output директорию ---
    fs::create_directories(cfg.output_dir);

    // --- CUDA stream ---
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ─── 1. Источник ────────────────────────────────────────────────────────
    printf("[1/5] Building source field...\n");
    ComplexField src = make_source_field(cfg);
    save_plane(src, "1_source", cfg, 4);

    // ─── 2. Коллимирующая линза (CrossLens) → пространство → коллиматор ───
    printf("[2/5] Propagating to collimator...\n");

    // Строим операторы
    Operator op_collim_x = Operator::allocate(cfg.source_plane.nx, cfg.collim_plane.nx);
    Operator op_collim_y = Operator::allocate(cfg.collim_plane.ny, cfg.source_plane.ny);

    build_sinc_operator_x(op_collim_x, cfg.source_plane, cfg.collim_plane, cfg, stream);
    build_sinc_operator_y(op_collim_y, cfg.source_plane, cfg.collim_plane, cfg, stream);

    ComplexField collim = ComplexField::allocate(cfg.collim_plane);
    apply_propagator(src, collim, op_collim_x, op_collim_y, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    save_plane(collim, "2_collimator", cfg, 2);

    op_collim_x.free(); op_collim_y.free();

    // ─── 3. Цилиндрическая линза fan-out ────────────────────────────────────
    printf("[3/5] Cylindrical lens fan-out...\n");

    // Применяем оператор цилиндрической линзы к коллимированному пучку
    Operator op_cyl_x = Operator::allocate(cfg.cyl_plane.nx, cfg.cyl_plane.nx);
    Operator op_cyl_y = Operator::allocate(cfg.cyl_plane.ny, cfg.cyl_plane.ny);
    build_cylind_lens_x(op_cyl_x, cfg.cyl_plane, cfg, stream);
    build_cylind_lens_y(op_cyl_y, cfg.cyl_plane, cfg, stream);

    ComplexField after_cyl = ComplexField::allocate(cfg.cyl_plane);
    apply_propagator(collim, after_cyl, op_cyl_x, op_cyl_y, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    save_plane(after_cyl, "3_cyl_lens", cfg, 2);

    op_cyl_x.free(); op_cyl_y.free();
    collim.free();

    // ─── 4. SLM ─────────────────────────────────────────────────────────────
    printf("[4/5] Applying SLM mask...\n");

    // Распространяем до плоскости SLM
    Plane sp = cfg.slm_plane();
    Operator op_slm_x = Operator::allocate(cfg.cyl_plane.nx, sp.nx);
    Operator op_slm_y = Operator::allocate(sp.ny, cfg.cyl_plane.ny);
    build_sinc_operator_x(op_slm_x, cfg.cyl_plane, sp, cfg, stream);
    build_sinc_operator_y(op_slm_y, cfg.cyl_plane, sp, cfg, stream);

    ComplexField slm_field = ComplexField::allocate(sp);
    apply_propagator(after_cyl, slm_field, op_slm_x, op_slm_y, stream);
    after_cyl.free();
    op_slm_x.free(); op_slm_y.free();

    // Умножаем на маску SLM
    ComplexField slm_mask = make_slm_mask(cfg);
    apply_slm(slm_field, slm_mask, stream);
    slm_mask.free();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    save_plane(slm_field, "4_slm", cfg, 8);   // scale=8 т.к. SLM маленький

    // ─── 5. Детекторы (fan-in) ──────────────────────────────────────────────
    printf("[5/5] Fan-in to detectors...\n");

    // Распространяем от SLM до плоскости детекторов
    Operator op_det_x = Operator::allocate(sp.nx, cfg.detector_plane.nx);
    Operator op_det_y = Operator::allocate(cfg.detector_plane.ny, sp.ny);
    build_sinc_operator_x(op_det_x, sp, cfg.detector_plane, cfg, stream);
    build_sinc_operator_y(op_det_y, sp, cfg.detector_plane, cfg, stream);

    ComplexField det_field = ComplexField::allocate(cfg.detector_plane);
    apply_propagator(slm_field, det_field, op_det_x, op_det_y, stream);
    slm_field.free();
    op_det_x.free(); op_det_y.free();

    CUDA_CHECK(cudaStreamSynchronize(stream));
    save_plane(det_field, "5_detector", cfg, 4);

    // Суммируем интенсивность по детекторам
    float* det_vals_gpu;
    CUDA_CHECK(cudaMalloc(&det_vals_gpu, sizeof(float) * cfg.num_detectors));
    fan_in(det_field, det_vals_gpu, cfg.num_detectors, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> det_vals(cfg.num_detectors);
    CUDA_CHECK(cudaMemcpy(det_vals.data(), det_vals_gpu,
                          sizeof(float) * cfg.num_detectors,
                          cudaMemcpyDeviceToHost));
    cudaFree(det_vals_gpu);
    det_field.free();

    // ─── Вывод результатов ──────────────────────────────────────────────────
    printf("\n=== Detector intensities ===\n");
    for (int i = 0; i < cfg.num_detectors; ++i) {
        printf("  PD[%d] = %.4e\n", i, det_vals[i]);
    }

    printf("\nDone. Results saved to: %s/\n", cfg.output_dir.c_str());

    CUDA_CHECK(cudaStreamDestroy(stream));
    src.free();
    return 0;
}