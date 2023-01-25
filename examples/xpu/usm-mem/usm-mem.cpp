/*
  Copyright (c) 2022-2023, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
