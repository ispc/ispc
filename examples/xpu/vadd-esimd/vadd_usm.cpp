/*
  Copyright (c) 2021-2023, Intel Corporation
*/

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

using ptr = float *;
static inline constexpr unsigned VL = 8;

extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void do_add_ispc(ptr A, ptr B, ptr C) SYCL_ESIMD_FUNCTION;

int main(void) {
    constexpr unsigned Size = 1024;
    constexpr unsigned GroupSize = 8;

    queue q(sycl::gpu_selector_v);

    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();
    float *A = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
    float *B = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
    float *C = static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

    for (unsigned i = 0; i < Size; ++i) {
        A[i] = B[i] = i;
    }

    // We need that many workitems. Each processes VL elements of data.
    cl::sycl::range<1> GlobalRange{Size / VL};
    // Number of workitems in each workgroup.
    cl::sycl::range<1> LocalRange{GroupSize};

    cl::sycl::nd_range<1> Range(GlobalRange, LocalRange);

    try {
        auto e = q.submit([&](handler &cgh) {
            cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                int i = ndi.get_global_id(0);
                do_add_ispc(ptr{A + i * VL}, B + i * VL, ptr{C + i * VL});
            });
        });
        e.wait();
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';

        free(A, ctxt);
        free(B, ctxt);
        free(C, ctxt);

        return 1;
    }

    int err_cnt = 0;

    for (unsigned i = 0; i < Size; ++i) {
        if (A[i] + B[i] != C[i]) {
            if (++err_cnt < 10) {
                std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i] << " + " << B[i] << "\n";
            }
        }
    }

    free(A, ctxt);
    free(B, ctxt);
    free(C, ctxt);

    if (err_cnt > 0) {
        std::cout << "  pass rate: " << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% (" << (Size - err_cnt)
                  << "/" << Size << ")\n";
        std::cout << "FAILED\n";
        return 1;
    }

    std::cout << "Passed\n";
    return 0;
}
