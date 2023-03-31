/*
  Copyright (c) 2020-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

// ispcrt
#include "ispcrt.hpp"

std::ostream &operator<<(std::ostream &os, const ISPCRTDeviceType dt) {
    switch (dt) {
    case ISPCRT_DEVICE_TYPE_AUTO:
        os << "Auto";
        break;
    case ISPCRT_DEVICE_TYPE_GPU:
        os << "GPU";
        break;
    case ISPCRT_DEVICE_TYPE_CPU:
        os << "CPU";
        break;
    default:
        break;
    }
    return os;
}

struct Parameters {
    float *vec;
    int count;
};

void simple_CPU_validation(const ispcrt::SharedVector<float>& vin, std::vector<float> &vgold, const unsigned int SIZE) {
    for (unsigned int i = 0; i < SIZE; i++) {
        float v = vin[i];
        if (v < 3.)
            v = v * v;
        else
            v = std::sqrt(v);

        vgold[i] = v;
    }
}

#define EPSILON 0.01f
bool validate_result(const ispcrt::SharedVector<float>& vec, const std::vector<float>& vgold, const unsigned int SIZE) {
    bool bValid = true;
    for (unsigned int i = 0; i < SIZE; i++) {
        float delta = (float)fabs(vgold[i] - vec[i]);
        if (delta > EPSILON) {
            std::cout << "Validation failed on i=" << i << ": vec[i] = " << vec[i] << ", but " << vgold[i]
                      << " was expected\n";
            bValid = false;
        }
    }
    return bValid;
}

static int run(const ISPCRTDeviceType device_type, const unsigned int SIZE) {
    ispcrt::Context context(device_type);

    // Create allocator for USM memory
    ispcrt::SharedMemoryAllocator<float> sma(context);

    // Allocate a vector of floats in the shared memory
    ispcrt::SharedVector<float> vec(SIZE, sma);

    // Golden results can be kept in host memory
    std::vector<float> vgold(SIZE);

    // Setup parameters structure - in shared memory
    ispcrt::Array<Parameters, ispcrt::AllocType::Shared> p(context);
    auto pp = p.sharedPtr();

    // Pass data pointers to the device
    pp->vec = vec.data();
    pp->count = SIZE;

    // Create device from context
    ispcrt::Device device(context);
    // Create module and kernel to execute
    ispcrt::Module module(device, "xe_simple-usm");
    ispcrt::Kernel kernel(device, module, "simple_ispc");

    // Create task queue and execute kernel
    ispcrt::TaskQueue queue(device);

    std::generate(vec.begin(), vec.end(), [i = 0]() mutable { return i++; });

    // Calculate gold result
    simple_CPU_validation(vec, vgold, SIZE);

    // No need to explicitly copy data to or from device

    // Launch the kernel on the device using 1 thread
    queue.launch(kernel, p, 1);

    // Execute queue and sync
    queue.sync();

    std::cout << "Executed on: " << device_type << '\n' << std::setprecision(6) << std::fixed;

    // Check and print result
    bool bValid = validate_result(vec, vgold, SIZE);
    if (bValid) {
        for (int i = 0; i < SIZE; i++) {
            std::cout << i << ": simple(" << i << ") = " << vec[i] << '\n';
        }
        return 0;
    }
    return -1;
}

void usage(const char *p) {
    std::cout << "Usage:\n";
    std::cout << p << " --cpu | --gpu | -h\n";
}

int main(int argc, char *argv[]) {
    std::ios_base::fmtflags f(std::cout.flags());

    constexpr unsigned int SIZE = 16;

    // Run on CPU by default
    ISPCRTDeviceType device_type = ISPCRT_DEVICE_TYPE_AUTO;

    if (argc > 2 || (argc == 2 && std::string(argv[1]) == "-h")) {
        usage(argv[0]);
        return -1;
    }

    if (argc == 2) {
        std::string dev_param = argv[1];
        if (dev_param == "--cpu") {
            device_type = ISPCRT_DEVICE_TYPE_CPU;
        } else if (dev_param == "--gpu") {
            device_type = ISPCRT_DEVICE_TYPE_GPU;
        } else {
            usage(argv[0]);
            return -1;
        }
    }

    int success = run(device_type, SIZE);
    std::cout.flags(f);
    return success;
}
