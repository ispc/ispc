/*
  Copyright (c) 2021, Intel Corporation
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
    float *vin;
    float *vout;
    int count;
};

void simple_CPU_validation(std::vector<float> vin, std::vector<float> &vgold, const unsigned int SIZE) {
    for (unsigned int i = 0; i < SIZE; i++) {
        float v = vin[i];
        if (v < 3.)
            v = v * v;
        else
            v = std::sqrt(v);

        vgold[i] = v * 2;
    }
}

#define EPSILON 0.01f
bool validate_result(std::vector<float> vout, std::vector<float> vgold, const unsigned int SIZE) {
    bool bValid = true;
    for (unsigned int i = 0; i < SIZE; i++) {
        float delta = (float)fabs(vgold[i] - vout[i]);
        if (delta > EPSILON) {
            std::cout << "Validation failed on i=" << i << ": vout[i] = " << vout[i] << ", but " << vgold[i]
                      << " was expected\n";
            bValid = false;
        }
    }
    return bValid;
}

static int run(const ISPCRTDeviceType device_type, const unsigned int SIZE) {
    std::vector<float> vin(SIZE);
    std::vector<float> vout(SIZE);
    std::vector<float> vgold(SIZE);

    ispcrt::Device device(device_type);

    // Setup input array
    ispcrt::Array<float> vin_dev(device, vin);

    // Setup output array
    ispcrt::Array<float> vout_dev(device, vout);

    // Setup parameters structure
    Parameters p;

    p.vin = vin_dev.devicePtr();
    p.vout = vout_dev.devicePtr();
    p.count = SIZE;

    auto p_dev = ispcrt::Array<Parameters>(device, p);

    // Create module and kernel to execute
    ispcrt::Module module(device, "simple-esimd_ispc2esimd");
    ispcrt::Kernel kernel(device, module, "simple_ispc");

    // Create task queue and execute kernel
    ispcrt::TaskQueue queue(device);

    std::generate(vin.begin(), vin.end(), [i = 0]() mutable { return i++; });

    // Calculate gold result
    simple_CPU_validation(vin, vgold, SIZE);

    // ispcrt::Array objects which used as inputs for ISPC kernel should be
    // explicitly copied to device from host
    queue.copyToDevice(p_dev);
    queue.copyToDevice(vin_dev);

    // Launch the kernel on the device using 1 thread
    queue.launch(kernel, p_dev, 1);

    // ispcrt::Array objects which used as outputs of ISPC kernel should be
    // explicitly copied to host from device
    queue.copyToHost(vout_dev);

    // Execute queue and sync
    queue.sync();

    std::cout << "Executed on: " << device_type << '\n' << std::setprecision(6) << std::fixed;

    // Check and print result
    bool bValid = validate_result(vout, vgold, SIZE);
    if (bValid) {
        for (int i = 0; i < SIZE; i++) {
            std::cout << i << ": simple(" << vin[i] << ") = " << vout[i] << '\n';
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
    return success;
}
