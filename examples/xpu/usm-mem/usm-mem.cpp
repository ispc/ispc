/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string.h>

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

static int run_same_mem(const ISPCRTDeviceType device_type, const unsigned int SIZE) {
    ispcrt::Context context(device_type);
    ispcrt::Device device(context);
    ispcrt::Module module(device, "xe_usm-mem");
    ispcrt::Kernel kernel(device, module, "usm_mem");

    ispcrt::SharedMemoryAllocator<float> sma(context, ispcrt::SharedMemoryUsageHint::HostWriteDeviceRead);

    for (int i = 0; i < 1<<6; i++) {
        ispcrt::SharedVector<float> vec(SIZE, sma);
        std::vector<float> vgold(SIZE);

        ispcrt::Array<Parameters, ispcrt::AllocType::Shared> p(context, ispcrt::SharedMemoryUsageHint::HostWriteDeviceRead);
        auto pp = p.sharedPtr();

        // Pass data pointers to the device
        pp->vec = vec.data();
        pp->count = SIZE;

        ispcrt::TaskQueue queue(device);

        std::generate(vec.begin(), vec.end(), [i = 0]() mutable { return i++; });
        simple_CPU_validation(vec, vgold, SIZE);

        queue.launch(kernel, p, 1);
        queue.sync();

        std::cout << i << " (same) Executed on: " << device_type << '\n';

        // Check and print result
        bool bValid = validate_result(vec, vgold, SIZE);
        if (!bValid) {
            std::cout << "Not valid" << std::endl;
            return -1;
        }
    }
    return 0;
}

int run_many_small(const ISPCRTDeviceType device_type) {
    ispcrt::Context context(device_type);

    ispcrt::SharedMemoryAllocator<char> sma(context, ispcrt::SharedMemoryUsageHint::HostWriteDeviceRead);

    for (int i = 0; i < 1<<6; i++) {
        sma.allocate(5);
        std::cout << i << " (small) Executed on: " << device_type << '\n';
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int rc = 0;
    constexpr unsigned int SIZE = 16;

    ISPCRTDeviceType device_type = ISPCRT_DEVICE_TYPE_GPU;

    if (argc != 2) {
        printf("test 1|2\n");
        return 1;
    }

    if (!strncmp(argv[1], "1\0", 2)) {
        rc = run_same_mem(device_type, SIZE);
        return rc;
    }

    if (!strncmp(argv[1], "2\0", 2)) {
        rc = run_many_small(device_type);
        return rc;
    }
    return 2;
}
