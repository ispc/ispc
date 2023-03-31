/*
  Copyright (c) 2020-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/
#ifdef _MSC_VER
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
// preventing MSVC fopen() deprecation complaints
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "common_helpers.h"
#include "timing.h"

// ispcrt
#include "ispcrt.hpp"


#define CORRECTNESS_THRESHOLD 0.0002
#define WIDTH 768
#define HEIGHT 768
#define SZ WIDTH *HEIGHT
#define TIMEOUT (40 * 1000)

extern void noise_serial(float x0, float y0, float x1, float y1, int width, int height, float output[]);

using namespace hostutil;

struct Parameters {
    float x0;
    float y0;
    float x1;
    float y1;
    int width;
    int height;
    float *output{nullptr};
};

static int run(int niter, int gx, int gy) {
    std::cout.setf(std::ios::unitbuf);
    const unsigned int height = HEIGHT;
    const unsigned int width = WIDTH;

    const float x0 = -10;
    const float y0 = -10;
    const float x1 = 10;
    const float y1 = 10;

    std::vector<float> buf(SZ);
    std::vector<float> gold(SZ);

    auto run_kernel = [&](ISPCRTDeviceType type) {
        ispcrt::Device device(type);

        // Setup output array
        ispcrt::Array<float> buf_dev(device, buf);

        // Setup parameters structure
        Parameters p;

        p.x0 = x0;
        p.y0 = y0;
        p.x1 = x1;
        p.y1 = y1;
        p.width = width;
        p.height = height;
        p.output = buf_dev.devicePtr();

        auto p_dev = ispcrt::Array<Parameters>(device, p);

        // Create module and kernel to execute
        ispcrt::Module module(device, "xe_noise");
        ispcrt::Kernel kernel(device, module, "noise_ispc");

        // Create task queue and execute kernel
        ispcrt::TaskQueue queue(device);
        double minCyclesISPC = 1e30;
        double kernelTicks = 1e30;
        const char *device_str = (type == ISPCRT_DEVICE_TYPE_GPU) ? "GPU" : "CPU";
        std::fill(buf.begin(), buf.end(), 0);
        for (unsigned int i = 0; i < niter; i++) {
            reset_and_start_timer();
            queue.copyToDevice(p_dev);
            auto res = queue.launch(kernel, p_dev, gx, gy);
            queue.copyToHost(buf_dev);
            queue.sync();
            if (res.valid()) {
                kernelTicks = res.time() * 1e-6;
            }
            double mcycles = get_elapsed_mcycles();
            // Print resulting time
            printf("@time of %s run:\t\t\t[%.3f] milliseconds\n", device_str, kernelTicks);
            printf("@time of %s run:\t\t\t[%.3f] million cycles\n", device_str, mcycles);
            minCyclesISPC = std::min(minCyclesISPC, mcycles);
        }
        printf("[noise ISPC %s]:\t\t[%.3f] million cycles (%d x %d image)\n", device_str, minCyclesISPC, width, height);
    };

    run_kernel(ISPCRT_DEVICE_TYPE_CPU);
    run_kernel(ISPCRT_DEVICE_TYPE_GPU);

    double minCyclesSerial = 1e30;
    std::fill(gold.begin(), gold.end(), 0);
    for (unsigned int i = 0; i < niter; i++) {
        reset_and_start_timer();
        auto wct = std::chrono::system_clock::now();
        noise_serial(x0, y0, x1, y1, width, height, gold.data());
        double mcycles = get_elapsed_mcycles();
        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(dur);

        // Print resulting time
        printf("@time of serial run:\t\t\t[%ld] milliseconds\n", secs.count());
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", mcycles);
        minCyclesSerial = std::min(minCyclesSerial, mcycles);
    }

    printf("[noise serial]:\t\t[%.3f] million cycles (%d x %d image)\n", minCyclesSerial, width, height);

    // Result check
    bool pass = true;
    double err = 0.0;
    double max_err = 0.0;

    int i = 0;
    for (; i < width * height; i++) {
        err = std::fabs(buf.at(i) - gold.at(i));
        max_err = std::max(err, max_err);
        if (err > CORRECTNESS_THRESHOLD) {
            pass = false;
            break;
        }
    }
    if (!pass) {
        std::cout << "Mismatch on " << i << "th value." << std::endl;
        std::cout << "Was " << buf.at(i) << ", should be " << gold.at(i) << std::endl;
    } else {
        std::cout << "No issues found, max error:" << max_err << std::endl;
    }

    return (pass) ? 0 : 1;
}

static void usage() {
    fprintf(stderr, "usage: noise [niterations] [group threads width] [group threads height]\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    int niterations = 1;
    int gx = 1, gy = 8;
    if (argc == 4) {
        niterations = atoi(argv[1]);
        gx = atoi(argv[2]);
        gy = atoi(argv[3]);
    }
    if (niterations < 1 || gx < 1 || gy < 1) {
        usage();
    }
    int success = 0;

    std::cout << "Running test with " << niterations << " iterations of ISPC on Xe and CPU using " << gx << " * " << gy
              << " threads." << std::endl;
    success = run(niterations, gx, gy);

    return success;
}
