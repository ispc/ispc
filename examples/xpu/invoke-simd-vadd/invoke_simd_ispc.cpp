// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/invoke_simd.hpp>
#include <sycl/sycl.hpp>

#include <functional>
#include <iostream>
#include <type_traits>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

constexpr int VL = 16;

// ISPC function declaration must have SYCL_EXTERNAL and SYCL_ESIMD_FUNCTION attributes.
extern "C" [[intel::device_indirectly_callable]] SYCL_EXTERNAL
    simd<float, VL> __regcall ISPC_CALLEE(float *A, simd<float, VL> b, int i) SYCL_ESIMD_FUNCTION;

int main(void) {
    constexpr unsigned Size = 1024;
    constexpr unsigned GroupSize = 4 * VL;

    queue q(sycl::gpu_selector_v);

    auto dev = q.get_device();

    std::cout << "Running on " << dev.get_info<sycl::info::device::name>() << "\n";

    auto ctxt = q.get_context();
    float *A = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
    float *B = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
    float *C = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

    for (unsigned i = 0; i < Size; ++i) {
        A[i] = i;
        B[i] = A[i] + 1;
        C[i] = -1;
    }

    sycl::range<1> GlobalRange{Size};
    // Number of workitems in each workgroup.
    sycl::range<1> LocalRange{GroupSize};
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    try {
        auto e = q.submit([&](handler &cgh) {
            cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) [[intel::reqd_sub_group_size(VL)]] {
                sub_group sg = ndi.get_sub_group();
                group<1> g = ndi.get_group();
                uint32_t i = sg.get_group_linear_id() * VL + g.get_group_linear_id() * GroupSize;
                uint32_t wi_id = i + sg.get_local_id();
                // According to invoke_simd spec the parameters will have the following types from ISPC perspective:
                // uniform{A} - uniform float* uniform A
                // B[wi_id] - varying float B
                // uniform{i} - uniform int32 i
                float res = invoke_simd(sg, ISPC_CALLEE, uniform{A}, B[wi_id], uniform{i});
                C[wi_id] = res;
            });
        });
        e.wait();
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        sycl::free(A, q);
        sycl::free(B, q);
        sycl::free(C, q);
        return -1;
    }

    int err_cnt = 0;
    for (unsigned i = 0; i < Size; ++i) {
        if (i % 2 == 0) {
            if (A[i] + B[i] != C[i]) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i] << " + " << B[i] << "\n";
                }
            }
        } else {
            if (A[i] - B[i] != C[i]) {
                if (++err_cnt < 10) {
                    std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i] << " - " << B[i] << "\n";
                }
            }
        }
    }
    if (err_cnt > 0) {
        std::cout << "  pass rate: " << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% (" << (Size - err_cnt)
                  << "/" << Size << ")\n";
    }

    std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
    return err_cnt > 0 ? 1 : 0;
}
