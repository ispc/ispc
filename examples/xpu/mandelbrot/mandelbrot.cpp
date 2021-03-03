/*
  Copyright (c) 2010-2021, Intel Corporation
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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
#endif

#include "timing.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string.h>

// ispcrt
#include "ispcrt.hpp"

extern void mandelbrot_cpp(float x0, float y0, float x1, float y1, int width, int height, int maxIterations,
                           int output[]);

struct Parameters {
    float x0;
    float dx;
    float y0;
    float dy;
    int width;
    int height;
    int tile_size{32};
    int maxIterations;
    int *output{nullptr};
};

/* Write a PPM image file with the image of the Mandelbrot set */
static void writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
    if (!fp) {
        perror(fn);
        exit(1);
    }
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255\n");
    for (int i = 0; i < width * height; ++i) {
        // Map the iteration count to colors by just alternating between
        // two greys.
        char c = (buf[i] & 0x1) ? (char)240 : 20;
        for (int j = 0; j < 3; ++j)
            fputc(c, fp);
    }
    fclose(fp);
    printf("Wrote image file %s\n", fn);
}

static int run(unsigned int width, unsigned int height, unsigned int test_iterations[]) {
    std::cout.setf(std::ios::unitbuf);

    int maxIterations = 512;
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;

    std::vector<int> buf(width * height);
    auto run_kernel = [&](ISPCRTDeviceType type) {
        ispcrt::Device device(type);

        // Setup output array
        ispcrt::Array<int> buf_dev(device, buf);

        // Setup parameters structure
        Parameters p;

        p.dx = (x1 - x0) / width;
        p.dy = (y1 - y0) / height;
        p.x0 = x0;
        p.y0 = y0;
        p.width = width;
        p.height = height;
        p.maxIterations = maxIterations;
        p.output = buf_dev.devicePtr();

        auto p_dev = ispcrt::Array<Parameters>(device, p);

        // Create module and kernel to execute
        ispcrt::Module module(device, "genx_mandelbrot");
        ispcrt::Kernel kernel(device, module, "mandelbrot_tile");

        // Create task queue and execute kernel
        ispcrt::TaskQueue queue(device);

        std::fill(buf.begin(), buf.end(), 0);
        double minTimeISPC = 1e30;
        double kernelTicks = 1e30;
        const char *device_str = (type == ISPCRT_DEVICE_TYPE_GPU) ? "GPU" : "CPU";
        std::fill(buf.begin(), buf.end(), 0);
        for (unsigned int i = 0; i < test_iterations[0]; ++i) {
            reset_and_start_timer();
            queue.copyToDevice(p_dev);
            queue.barrier();
            queue.submit();
            queue.sync();
            auto res = queue.launch(kernel, p_dev, width / p.tile_size, height / p.tile_size);
            queue.barrier();
            queue.copyToHost(buf_dev);
            queue.barrier();
            queue.sync();
            if (res.valid()) {
                kernelTicks = res.time() * 1e-6;
            }
            double mcycles = get_elapsed_mcycles();

            // Print resulting time
            printf("@time of %s run:\t\t\t[%.3f] milliseconds\n", device_str, kernelTicks);
            printf("@time of %s run:\t\t\t[%.3f] million cycles\n", device_str, mcycles);
            minTimeISPC = std::min(minTimeISPC, mcycles);
        }
        printf("[mandelbrot ISPC %s]:\t\t[%.3f] million cycles (%d x %d image)\n", device_str, minTimeISPC, width,
               height);
    };
    run_kernel(ISPCRT_DEVICE_TYPE_CPU);
    writePPM(buf.data(), width, height, "mandelbrot-ispc-cpu.ppm");

    run_kernel(ISPCRT_DEVICE_TYPE_GPU);
    writePPM(buf.data(), width, height, "mandelbrot-ispc-gpu.ppm");

    double minTimeSerial = 1e30;
    std::fill(buf.begin(), buf.end(), 0);
    for (unsigned int i = 0; i < test_iterations[1]; ++i) {
        reset_and_start_timer();
        auto wct = std::chrono::system_clock::now();
        mandelbrot_cpp(x0, y0, x1, y1, width, height, maxIterations, buf.data());
        double mcycles = get_elapsed_mcycles();
        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(dur);

        // Print resulting time
        printf("@time of serial run:\t\t\t[%ld] milliseconds\n", secs.count());
        printf("@time of serial run:\t\t\t[%.3f] million cycles\n", mcycles);
        minTimeSerial = std::min(minTimeSerial, mcycles);
    }
    printf("[mandelbrot serial]:\t\t[%.3f] million cycles (%d x %d image)\n", minTimeSerial, width, height);
    writePPM(buf.data(), width, height, "mandelbrot-cpp-serial.ppm");

    return 0;
}

static void usage() {
    fprintf(stderr, "usage: mandelbrot [--scale=<factor>] [tasks iterations] [serial iterations]\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {3, 1};
    unsigned int width = 1536;
    unsigned int height = 1024;

    if (argc > 1) {
        if (strncmp(argv[1], "--scale=", 8) == 0) {
            float scale = atof(argv[1] + 8);
            if (scale == 0.f)
                usage();
            width *= scale;
            height *= scale;
            // round up to multiples of 16
            width = (width + 0xf) & ~0xf;
            height = (height + 0xf) & ~0xf;
        }
    }
    if ((argc == 3) || (argc == 4)) {
        for (int i = 0; i < 2; i++) {
            test_iterations[i] = atoi(argv[argc - 2 + i]);
        }
    }
    int success = 0;
    std::cout << "Running test with " << test_iterations[0] << " iterations of ISPC on GEN and CPU and "
              << test_iterations[1] << " serial iterations on " << width << " * " << height << " size." << std::endl;
    success = run(width, height, test_iterations);

    return success;
}
