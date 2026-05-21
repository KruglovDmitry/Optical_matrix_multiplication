#pragma once
#include "field.hpp"
#include <string>

// ─── Fan-out / Fan-in ───────────────────────────────────────────────────────
// fan_out: размножает строку источника по всем строкам SLM
void fan_out(const ComplexField& src,
              ComplexField&       dst,
              cudaStream_t        stream = 0);

// apply_slm: поэлементное умножение поля на маску SLM
void apply_slm(ComplexField&       field,
                const ComplexField& mask,
                cudaStream_t        stream = 0);

// fan_in: суммирует интенсивность по строкам → значения детекторов
void fan_in(const ComplexField& field,
             float*              detectors_gpu,
             int                 num_detectors,
             cudaStream_t        stream = 0);

// ─── Интенсивность ──────────────────────────────────────────────────────────
void compute_intensity(const ComplexField& field,
                        IntensityMap&       out,
                        cudaStream_t        stream = 0);

void normalize_intensity(IntensityMap& map,
                          cudaStream_t  stream = 0);

void save_intensity_png(const IntensityMap& map,
                         const std::string&  path,
                         int                 scale_factor = 1,
                         cudaStream_t        stream = 0);