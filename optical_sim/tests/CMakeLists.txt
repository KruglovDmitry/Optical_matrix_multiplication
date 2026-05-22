cmake_minimum_required(VERSION 3.20)

add_executable(test_field test_field.cu
               ../kernels/intensity.cu)
target_include_directories(test_field PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_field PRIVATE CUDA::cudart)
add_test(NAME FieldAlloc COMMAND test_field)

add_executable(test_fresnel test_fresnel.cu
               ../kernels/fresnel.cu
               ../kernels/intensity.cu)
target_include_directories(test_fresnel PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_fresnel PRIVATE CUDA::cudart)
add_test(NAME FresnelPropagator COMMAND test_fresnel)