CUDA kernels → PBO (pixel buffer object) → EGL offscreen → PNG/stream
                                                              ↓
                                                    веб-интерфейс или SSH -X

# Структура проекта

optical_sim/
├── CMakeLists.txt
├── include/
│   ├── config.hpp          ← параметры системы (λ, f, SLM size, N источников)
│   ├── field.hpp           ← структура светового поля на плоскости
│   ├── propagator.hpp      ← базовый класс пропагаторов
│   └── renderer.hpp        ← OpenGL/EGL визуализация
│
├── src/
│   ├── main.cpp            ← точка входа, UI логика
│   └── renderer.cpp        ← EGL контекст, шейдеры
│
├── kernels/
│   ├── fresnel.cu          ← PropagatorSinc на CUDA
│   ├── lens.cu             ← CrossLens + CylindLens
│   ├── fan.cu              ← fan-out и fan-in
│   └── intensity.cu        ← |E|² + нормализация для рендера
│
└── shaders/
    ├── field.vert
    └── field.frag          ← colormap (например, inferno)

# Работа 

main.cpp
  │
  ├─ загружает SystemConfig (из файла или интерактивно)
  │
  ├─ SimulationPipeline::run()
  │    ├─ fan_out_kernel<<<>>>()      VCSEL → все строки SLM
  │    ├─ fresnel_kernel<<<>>>()      свободное пространство
  │    ├─ lens_kernel<<<>>>()         линзы
  │    ├─ slm_multiply<<<>>>()        умножение на матрицу W
  │    ├─ fan_in_kernel<<<>>>()       суммирование на PD
  │    └─ intensity_kernel<<<>>>()    |E|² для каждого слоя
  │
  └─ Renderer::draw(PlaneID)          выбираем какой слой смотреть

  ![alt text](image.png)