/*
  Copyright (c) 2021-2023, Intel Corporation
*/

#include <iostream>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

constexpr unsigned VL = 8;

extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void vmult(simd<float, VL> *a, simd<float, VL> *res, int factor, int n) {
    for (int i = 0; i < n / VL; ++i) {
        auto b = a[i];
        res[i] = b * factor;
    }
}
