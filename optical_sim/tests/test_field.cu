#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include "field.hpp"
#include "pipeline.hpp"

int main()
{
    printf("=== test_field ===\n");

    // 1. Аллокация
    Plane p = { 64, 64, 1e-6f, 1e-6f };
    ComplexField f = ComplexField::allocate(p);
    assert(f.data != nullptr);
    assert(f.nx == 64 && f.ny == 64);
    printf("[OK] ComplexField::allocate\n");

    // 2. Upload / download
    std::vector<cuComplex> host_in(64*64);
    for (int i = 0; i < 64*64; ++i)
        host_in[i] = make_cuComplex((float)i, -(float)i);

    f.upload(host_in.data());

    std::vector<cuComplex> host_out(64*64, make_cuComplex(0,0));
    f.download(host_out.data());

    for (int i = 0; i < 64*64; ++i) {
        assert(fabsf(host_out[i].x - host_in[i].x) < 1e-5f);
        assert(fabsf(host_out[i].y - host_in[i].y) < 1e-5f);
    }
    printf("[OK] Upload / Download roundtrip\n");

    // 3. IntensityMap
    IntensityMap imap = IntensityMap::allocate(64, 64);
    assert(imap.data != nullptr);

    compute_intensity(f, imap);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> ints(64*64);
    imap.download(ints.data());

    // |i - i*j|² = i² + i² = 2i²
    for (int i = 0; i < 64*64; ++i) {
        float expected = 2.f * (float)i * (float)i;
        assert(fabsf(ints[i] - expected) < 1.f);  // допуск на float
    }
    printf("[OK] compute_intensity\n");

    f.free();
    imap.free();

    printf("\n=== All tests passed ===\n");
    return 0;
}
