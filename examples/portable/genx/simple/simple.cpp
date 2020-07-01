/*
  Copyright (c) 2020, Intel Corporation
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

#include <iostream>
#include <iomanip>
#include <algorithm>

// ispcrt
#include "ispcrt.hpp"

struct Parameters {
    float *vin;
    float *vout;
    int    count;
};

static void run(const ISPCRTDeviceType device_type, const unsigned int SIZE) {
    std::vector<float> vin(SIZE);
    std::vector<float> vout(SIZE);

    ispcrt::Device device(device_type);

    // Setup input array
    ispcrt::Array<float> vin_dev(device, vin);

    // Setup output array
    ispcrt::Array<float> vout_dev(device, vout);

    // Setup parameters structure
    Parameters p;

    p.vin   = vin_dev.devicePtr();
    p.vout  = vout_dev.devicePtr();
    p.count = SIZE;

    auto p_dev = ispcrt::Array<Parameters>(device, p);

    // Create module and kernel to execute
    ispcrt::Module module(device, "genx_simple");
    ispcrt::Kernel kernel(device, module, "simple_ispc");

    // Create task queue and execute kernel
    ispcrt::TaskQueue queue(device);

    std::generate(vin.begin(), vin.end(), [i = 0] () mutable { return i++; });

    queue.copyToDevice(p_dev);
    queue.copyToDevice(vin_dev);
    queue.barrier();

    // Launch the kernel on the device
    queue.launch(kernel, p_dev, 1);
    queue.barrier();

    queue.copyToHost(vout_dev);
    queue.barrier();
    queue.sync();

    std::string device_str;

    switch (device_type)
    {
    case ISPCRT_DEVICE_TYPE_AUTO:
        device_str = "Auto";
        break;
    case ISPCRT_DEVICE_TYPE_GPU:
        device_str = "GPU";
        break;
    case ISPCRT_DEVICE_TYPE_CPU:
        device_str = "CPU";
        break;
    default:
        break;
    }

    std::cout << "Executed on: " << device_str << '\n';
    std::cout << std::setprecision(6) << std::fixed;
    for (int i = 0; i < SIZE; i++) {
        std::cout << i << ": simple(" << vin[i] << ") = " << vout[i] << '\n';
    }
}

void usage(const char *p) {
    std::cout << "Usage:\n";
    std::cout << p << " --cpu | --gpu | -h\n";
}

int main(int argc, char *argv[]) {
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

    run(device_type, SIZE);

    return 0;
}
