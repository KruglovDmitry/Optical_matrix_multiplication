cmake_minimum_required(VERSION 3.20)
project(optical_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- Найти CUDA ---
find_package(CUDAToolkit REQUIRED)

# --- Архитектуры: RTX 30/40xx = sm_86/sm_89, H100 = sm_90 ---
set(CMAKE_CUDA_ARCHITECTURES 86 89 90)

# --- Флаги компилятора ---
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --extended-lambda \
    --expt-relaxed-constexpr \
    -use_fast_math \
    --generate-line-info")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# --- Исходники ---
set(KERNEL_SOURCES
    kernels/fresnel.cu
    kernels/lens.cu
    kernels/intensity.cu
    kernels/fan.cu
)

set(SRC_SOURCES
    src/main.cpp
)

# --- Исполняемый файл ---
add_executable(optical_sim ${SRC_SOURCES} ${KERNEL_SOURCES})

target_include_directories(optical_sim PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(optical_sim PRIVATE
    CUDA::cudart
    CUDA::cufft        # пригодится позже для FFT-based propagation
)

# --- Тесты ---
enable_testing()
add_subdirectory(tests)