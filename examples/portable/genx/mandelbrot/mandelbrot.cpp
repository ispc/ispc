/*
  Copyright (c) 2010-2011, Intel Corporation
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

#include "L0_helpers.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string.h>

using namespace std;
using namespace hostutil;

extern void mandelbrot_cpp(float x0, float y0, float x1, float y1, int width, int height, int maxIterations,
                           int output[]);

/* Write a PPM image file with the image of the Mandelbrot set */
static void writePPM(int *buf, int width, int height, const char *fn) {
    FILE *fp = fopen(fn, "wb");
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

static void usage() {
    fprintf(stderr, "usage: mandelbrot [--scale=<factor>] [tasks iterations] [serial iterations]\n");
    exit(1);
}

static int run(unsigned int width, unsigned int height) {
    std::cout.setf(std::ios::unitbuf);
    int maxIterations = 512;
    int tileSize = 32;
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;
    int *buf = (int *)aligned_alloc(4096, width * height * sizeof(int));
    memset((void *)buf, 0, sizeof(int) * width * height);

    // L0 initialization
    ze_device_handle_t hDevice = nullptr;
    ze_module_handle_t hModule = nullptr;
    ze_driver_handle_t hDriver = nullptr;
    ze_command_queue_handle_t hCommandQueue = nullptr;
    L0InitContext(hDriver, hDevice, hModule, hCommandQueue, "mandelbrot_ispc.spv");

    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "mandelbrot_tile");

    // Alloc & copy
    ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT,
                                            0};
    void *dBuf = nullptr;
    L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, width * height * sizeof(int), width * height * sizeof(int),
                                        hDevice, &dBuf));

    // copy buffers to device
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, dBuf, buf, width * height * sizeof(int), nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

    // set kernel arguments
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(float), &x0));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(float), &dx));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(float), &y0));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(float), &dy));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &width));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &height));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 6, sizeof(int), &tileSize));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 7, sizeof(int), &maxIterations));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 8, width * height * sizeof(int), &dBuf));

    // thread space
    ze_group_count_t dispatchTraits = {width / (tileSize), height / (tileSize), 1};
    L0_SAFE_CALL(zeCommandListClose(hCommandList));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));
    L0_SAFE_CALL(zeCommandListReset(hCommandList));

    // launch
    auto wct = std::chrono::system_clock::now();
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &dispatchTraits, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

    L0_SAFE_CALL(zeCommandListClose(hCommandList));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));
    auto dur = (std::chrono::system_clock::now() - wct);
    auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
    std::cout << "Time is: " << secs.count() << " milliseconds" << std::endl;
    Timings(secs.count(), secs.count()).print(maxIterations);

    L0_SAFE_CALL(zeCommandListReset(hCommandList));

    // copy result to host
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, buf, dBuf, width * height * sizeof(int), nullptr));

    // dispatch & wait
    L0_SAFE_CALL(zeCommandListClose(hCommandList));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));

    writePPM(buf, width, height, "mandelbrot-ispc.ppm");

    double minSerial = 1e30;
    for (unsigned int i = 0; i < width * height; ++i)
        buf[i] = 0;
    mandelbrot_cpp(x0, y0, x1, y1, width, height, maxIterations, buf);

    writePPM(buf, width, height, "mandelbrot-cpp.ppm");

    free(buf);
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, dBuf));
    std::cout << "Passed!" << std::endl;
    return 0;
}

int main(int argc, char *argv[]) {
    static unsigned int test_iterations[] = {7, 1};
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

    success = run(width, height);
    std::cout << "SUCCESS" << success << std::endl;
    return success;
}
