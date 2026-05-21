struct SystemConfig {
    // Оптика
    float wavelength;       // λ, м
    float focal_distance;   // f
    float K;                // волновое число 2π/λ
    
    // Источники
    int   num_sources;      // N VCSEL
    float source_pitch;     // шаг решётки
    float divergence_angle; // θ ≈ 10–20°
    
    // SLM
    int   slm_rows;         // число строк (y₀..yₙ)
    int   slm_cols;         // пикселей по x
    float slm_pixel_size;
    
    // Детекторы
    int   num_detectors;    // число PD
};