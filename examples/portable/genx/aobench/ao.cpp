/*
  Copyright (c) 2010-2020, Intel Corporation
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
#define NOMINMAX
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)
// preventing MSVC fopen() deprecation complaints
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <malloc.h>
#endif
#include "L0_helpers.h"
#include "timing.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <string>
#include <sys/types.h>

#define NSUBSAMPLES 2
#define CORRECTNESS_THRESHOLD 0.01

using namespace hostutil;

extern void ao_serial(int w, int h, int nsubsamples, float image[]);

static unsigned int niterations;
static unsigned int width, height;
static unsigned char *img;
static float *fimg;

static unsigned char clamp(float f) {
    int i = (int)(f * 255.5);

    if (i < 0)
        i = 0;
    if (i > 255)
        i = 255;

    return (unsigned char)i;
}

static void savePPM(const char *fname, int w, int h, float *fimg) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            img[3 * (y * w + x) + 0] = clamp(fimg[3 * (y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 * (y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 * (y * w + x) + 2]);
        }
    }

    FILE *fp = fopen(fname, "wb");
    if (!fp) {
        perror(fname);
        exit(1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
    printf("Wrote image file %s\n", fname);
}

static int run() {
    std::cout.setf(std::ios::unitbuf);
    int nSubSamples = NSUBSAMPLES;
    img = new unsigned char[width * height * 3];
    fimg = (float *)aligned_alloc(4096, width * height * 3 * sizeof(float));

    // L0 initialization
    ze_device_handle_t hDevice = nullptr;
    ze_module_handle_t hModule = nullptr;
    ze_driver_handle_t hDriver = nullptr;
    ze_command_queue_handle_t hCommandQueue = nullptr;
    L0InitContext(hDriver, hDevice, hModule, hCommandQueue, "aobench_ispc.spv");

    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "ao_ispc");
    //
    // Run the ispc gpu code nIterations times, and report the minimum
    // time for any of them.
    //
    double minCyclesISPCGPU = 1e30;
    void *dFimg = nullptr;

    // thread space
    ze_group_count_t dispatchTraits = {1, height, 1};
    std::cout << "Set dispatchTraits.x=" << dispatchTraits.groupCountX
              << ", dispatchTraits.y=" << dispatchTraits.groupCountY << std::endl;
    for (unsigned int i = 0; i < niterations; i++) {
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);

        // Alloc & copy
        ze_device_mem_alloc_desc_t allocDesc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT,
                                                ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, 0};
        L0_SAFE_CALL(zeDriverAllocDeviceMem(hDriver, &allocDesc, width * height * 3 * sizeof(float),
                                            width * height * 3 * sizeof(float), hDevice, &dFimg));

        // copy buffers to device
        L0_SAFE_CALL(
            zeCommandListAppendMemoryCopy(hCommandList, dFimg, fimg, width * height * 3 * sizeof(float), nullptr));
        L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

        // set kernel arguments
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(int), &width));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(int), &height));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(int), &nSubSamples));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(dFimg), &dFimg));

        L0_SAFE_CALL(zeCommandListClose(hCommandList));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));
        L0_SAFE_CALL(zeCommandListReset(hCommandList));

        // launch
        reset_and_start_timer();
        auto wct = std::chrono::system_clock::now();

        L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &dispatchTraits, nullptr, 0, nullptr));
        L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));
        L0_SAFE_CALL(zeCommandListClose(hCommandList));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));

        double dt = get_elapsed_mcycles();
        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
        std::cout << "@time of GPU run:\t\t\t[" << secs.count() << "] milliseconds" << std::endl;
        printf("@time of GPU run:\t\t\t[%.3f] million cycles\n", dt);
        minCyclesISPCGPU = std::min(minCyclesISPCGPU, dt);

        L0_SAFE_CALL(zeCommandListReset(hCommandList));
        // copy result to host
        L0_SAFE_CALL(
            zeCommandListAppendMemoryCopy(hCommandList, fimg, dFimg, width * height * 3 * sizeof(float), nullptr));
        L0_SAFE_CALL(zeCommandListClose(hCommandList));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));
    }

    savePPM("ao-ispc-gpu.ppm", width, height, fimg);
    printf("[aobench ISPC GPU]:\t\t[%.3f] million cycles (%d x %d image)\n", minCyclesISPCGPU, width, height);

    //
    // Run the serial code nIterations times, and report the minimum
    // time for any of them.
    //
    double minCyclesSerial = 1e30;
    for (unsigned int i = 0; i < niterations; i++) {
        memset((void *)fimg, 0, sizeof(float) * width * height * 3);
        reset_and_start_timer();
        auto wct = std::chrono::system_clock::now();
        ao_serial(width, height, NSUBSAMPLES, fimg);
        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
        std::cout << "@time of CPU run:\t\t\t[" << secs.count() << "] milliseconds" << std::endl;
        double dt = get_elapsed_mcycles();
        printf("@time of CPU run:\t\t\t[%.3f] million cycles\n", dt);
        minCyclesSerial = std::min(minCyclesSerial, dt);
    }
    savePPM("ao-cpp-serial.ppm", width, height, fimg);

    printf("[aobench serial]:\t\t[%.3f] million cycles (%d x %d image)\n", minCyclesSerial, width, height);
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minCyclesSerial / minCyclesISPCGPU);
    free(fimg);
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, dFimg));
    printf("aobench PASSED!\n");
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("%s\n", argv[0]);
        printf("Usage: ao [num test iterations] [width] [height]\n");
        getchar();
        exit(-1);
    } else {
        niterations = atoi(argv[1]);
        width = atoi(argv[2]);
        height = atoi(argv[3]);
    }

    int success = 0;

    std::cout << "Running test with " << niterations << " iterations on " << width << " * " << height << " size."
              << std::endl;
    success = run();

    return success;
}
