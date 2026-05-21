#pragma once
#include <cmath>
#include <array>
#include <string>

// ─────────────────────────────────────────────
//  Расчётная плоскость  (≈ ConfigDesignPlane)
// ─────────────────────────────────────────────
struct Plane {
    int   nx, ny;           // число пикселей по X и Y
    float dx, dy;           // размер пикселя [м]

    // Апертура [м]
    float aperture_x() const { return nx * dx; }
    float aperture_y() const { return ny * dy; }

    // Координата i-го пикселя по X (центрированная сетка, как в Python)
    // linspace(-L/2, L/2, n+1)[:n]  + L/(2n)
    __host__ __device__
    float coord_x(int i) const {
        float L = nx * dx;
        return -L * 0.5f + dx * (i + 0.5f);
    }

    __host__ __device__
    float coord_y(int j) const {
        float L = ny * dy;
        return -L * 0.5f + dy * (j + 0.5f);
    }
};

// ─────────────────────────────────────────────
//  Конфигурация всей оптической системы
// ─────────────────────────────────────────────
struct SystemConfig {

    // --- Оптика ---
    float wavelength  = 850e-9f;            // λ VCSEL, м
    float distance    = 0.03f;              // f, расстояние между плоскостями, м
    float K() const { return 2.f * M_PI / wavelength; }

    // --- Массив источников (VCSEL) ---
    int   num_sources   = 6;               // N источников (строки вектора x)
    float source_pitch  = 50e-6f;          // шаг решётки VCSEL [м]
    float divergence    = 15.f;            // полуугол расходимости [°]

    // --- Плоскости системы ---
    //  SOURCE → COLLIMATOR → CYL_LENS → SLM → FAN_IN → DETECTOR

    Plane source_plane   = { 128, 6,    5e-6f, 50e-6f };
    //                       nx   ny   dx      dy
    //  6 источников по Y, 128 пикселей по X для одного пучка

    Plane collim_plane   = { 256, 256, 3.6e-6f, 3.6e-6f };

    // Цилиндрическая линза fan-out: один источник → все строки SLM
    Plane cyl_plane      = { 256, 256, 3.6e-6f, 3.6e-6f };

    // SLM (матрица W)
    int   slm_cols       = 256;
    int   slm_rows       = 6;              // = num_sources
    float slm_pixel_x    = 8e-6f;
    float slm_pixel_y    = 50e-6f;
    Plane slm_plane() const {
        return { slm_cols, slm_rows, slm_pixel_x, slm_pixel_y };
    }

    // Fan-in линза + детекторы
    int   num_detectors  = 6;
    Plane detector_plane = { 256, 6, 3.6e-6f, 50e-6f };

    // --- Вывод ---
    std::string output_dir = "./output";   // куда сохранять PNG
};

// ─────────────────────────────────────────────
//  Идентификатор плоскости для визуализации
// ─────────────────────────────────────────────
enum class PlaneID : int {
    SOURCE      = 0,
    COLLIMATOR  = 1,
    CYL_LENS    = 2,
    SLM         = 3,
    FAN_IN      = 4,
    DETECTOR    = 5,
    COUNT       = 6
};

inline const char* plane_name(PlaneID id) {
    static const char* names[] = {
        "source", "collimator", "cyl_lens", "slm", "fan_in", "detector"
    };
    return names[static_cast<int>(id)];
}