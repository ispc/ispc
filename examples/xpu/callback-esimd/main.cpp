/*
  Copyright (c) 2021-2023, Intel Corporation
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
    uint8_t *objects;
    float *output;
};

static int run(const ISPCRTDeviceType device_type) {
    ispcrt::Device device(device_type);

    // 32 bytes for 2 objects, each is 16
    std::vector<uint8_t> objects(2 * 16);

    std::vector<float> output(2);
    output[0] = 0;
    output[1] = 0;

    ispcrt::Array<uint8_t> objects_dev(device, objects);
    ispcrt::Array<float> output_dev(device, output);

    // Setup parameters structure
    Parameters p;

    p.objects = objects_dev.devicePtr();
    p.output = output_dev.devicePtr();

    auto p_dev = ispcrt::Array<Parameters>(device, p);

    // Create module and kernel to execute
    ispcrt::Module module(device, "callback-esimd_ispc2esimd");
    ispcrt::Kernel make_objects_kernel(device, module, "make_objects");
    ispcrt::Kernel call_objects_kernel(device, module, "call_objects");

    // Create task queue and execute kernel
    ispcrt::TaskQueue queue(device);

    // ispcrt::Array objects which used as inputs for ISPC kernel should be
    // explicitly copied to device from host
    queue.copyToDevice(p_dev);

    queue.launch(make_objects_kernel, p_dev, 1);
    queue.barrier();

    // Launch the kernel on the device using 1 thread
    queue.launch(call_objects_kernel, p_dev, 1);

    // ispcrt::Array objects which used as outputs of ISPC kernel should be
    // explicitly copied to host from device
    queue.copyToHost(output_dev);

    // Execute queue and sync
    queue.sync();

    std::cout << "Executed on: " << device_type << '\n' << std::setprecision(6) << std::fixed;

    // Check and print result
    bool bValid = output[0] == -1 && output[1] == -2;
    if (bValid) {
        std::cout << "Function was called successfully, output:\n";
        for (size_t i = 0; i < output.size(); ++i) {
            std::cout << "output[" << i << "] = " << output[i] << "\n";
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

    int success = run(device_type);
    return success;
}

