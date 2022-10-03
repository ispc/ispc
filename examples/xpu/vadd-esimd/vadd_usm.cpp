/*
  Copyright (c) 2021-2022, Intel Corporation
*/

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

using ptr = float *;
static inline constexpr unsigned VL = 8;

extern "C" SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void do_add_ispc(ptr A, ptr B, ptr C) SYCL_ESIMD_FUNCTION;

namespace esimd_test {

// This is the class provided to SYCL runtime by the application to decide
// on which device to run, or whether to run at all.
// When selecting a device, SYCL runtime first takes (1) a selector provided by
// the program or a default one and (2) the set of all available devices. Then
// it passes each device to the '()' operator of the selector. Device, for
// which '()' returned the highest number, is selected. If a negative number
// was returned for all devices, then the selection process will cause an
// exception.
class ESIMDSelector : public device_selector {
    // Require GPU device unless HOST is requested in SYCL_DEVICE_FILTER env
    virtual int operator()(const device &device) const {
        if (const char *dev_filter = getenv("SYCL_DEVICE_FILTER")) {
            std::string filter_string(dev_filter);
            if (filter_string.find("gpu") != std::string::npos)
                return device.is_gpu() ? 1000 : -1;
            if (filter_string.find("host") != std::string::npos)
                return device.is_host() ? 1000 : -1;
            std::cerr << "Supported 'SYCL_DEVICE_FILTER' env var values are 'gpu' and "
                         "'host', '"
                      << filter_string << "' does not contain such substrings.\n";
            return -1;
        }
        // If "SYCL_DEVICE_FILTER" not defined, only allow gpu device
        return device.is_gpu() ? 1000 : -1;
    }
};

inline auto createExceptionHandler() {
    return [](exception_list l) {
        for (auto ep : l) {
            try {
                std::rethrow_exception(ep);
            } catch (cl::sycl::exception &e0) {
                std::cout << "sycl::exception: " << e0.what() << std::endl;
            } catch (std::exception &e) {
                std::cout << "std::exception: " << e.what() << std::endl;
            } catch (...) {
                std::cout << "generic exception\n";
            }
        }
    };
}

} // namespace esimd_test

int main(void) {
    constexpr unsigned Size = 1024;
    constexpr unsigned GroupSize = 8;

    queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

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
