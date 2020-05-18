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

#include "L0_helpers.h"
#include "timing.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

#define CORRECTNESS_THRESHOLD 0.0002
#define SZ 768 * 768
#define TIMEOUT (40 * 1000)

extern void noise_serial(float x0, float y0, float x1, float y1, int width, int height, float output[]);

using namespace hostutil;

PageAlignedArray<float, SZ> out, gold, result;

static int run(int niter, int gx, int gy) {

    std::cout.setf(std::ios::unitbuf);
    ze_device_handle_t hDevice = nullptr;
    ze_module_handle_t hModule = nullptr;
    ze_driver_handle_t hDriver = nullptr;
    ze_command_queue_handle_t hCommandQueue = nullptr;

    for (int i = 0; i < SZ; i++) {
        out.data[i] = -1;
        gold.data[i] = -1;
    }

#ifdef CMKERNEL
    L0InitContext(hDriver, hDevice, hModule, hCommandQueue, "noise_cm.spv");
#else
    L0InitContext(hDriver, hDevice, hModule, hCommandQueue, "noise_ispc.spv");
#endif

    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;

#ifdef CMKERNEL
    //    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "sgemm_kernel");
#else
    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "noise_ispc");
#endif

    const unsigned int height = 768;
    const unsigned int width = 768;

    const float x0 = -10;
    const float y0 = -10;
    const float x1 = 10;
    const float y1 = 10;
    double minCyclesISPCGPU = 1e30;
    // set grid size
    ze_group_count_t dispatchTraits = {(uint32_t)gx, (uint32_t)gy, 1};
    std::cout << "Set dispatchTraits.x=" << dispatchTraits.groupCountX
              << ", dispatchTraits.y=" << dispatchTraits.groupCountY << std::endl;
    for (unsigned int i = 0; i < niter; i++) {
        void *buf_ref = out.data;
        ze_device_mem_alloc_desc_t alloc_desc = {ZE_DEVICE_MEM_ALLOC_DESC_VERSION_CURRENT,
                                                 ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, 0};
        ze_host_mem_alloc_desc_t host_alloc_desc = {ZE_HOST_MEM_ALLOC_DESC_VERSION_CURRENT,
                                                    ZE_HOST_MEM_ALLOC_FLAG_DEFAULT};

        L0_SAFE_CALL(zeDriverAllocSharedMem(hDriver, &alloc_desc, &host_alloc_desc, SZ * sizeof(float), 0 /*align*/,
                                            hDevice, &buf_ref));

        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(float), &x0));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(float), &y0));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(float), &x1));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(float), &y1));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &width));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &height));
        L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 6, sizeof(buf_ref), &buf_ref));

        reset_and_start_timer();
        auto wct = std::chrono::system_clock::now();
        // launch
        L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &dispatchTraits, nullptr, 0, nullptr));
        L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

        // copy result to host
        void *res = result.data;
        L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, res, buf_ref, SZ * sizeof(float), nullptr));
        // dispatch & wait
        L0_SAFE_CALL(zeCommandListClose(hCommandList));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));

        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
        std::cout << "@time of GPU run:\t\t\t[" << secs.count() << "] milliseconds" << std::endl;
        double dt = get_elapsed_mcycles();
        printf("@time of GPU run:\t\t\t[%.3f] million cycles\n", dt);
        minCyclesISPCGPU = std::min(minCyclesISPCGPU, dt);
        L0_SAFE_CALL(zeDriverFreeMem(hDriver, buf_ref));
    }
    printf("[noise ISPC GPU]:\t\t[%.3f] million cycles (%d x %d image)\n", minCyclesISPCGPU, width, height);

    double minCyclesSerial = 1e30;
    for (unsigned int i = 0; i < niter; i++) {
        reset_and_start_timer();
        auto wct = std::chrono::system_clock::now();
        noise_serial(x0, y0, x1, y1, width, height, gold.data);
        auto dur = (std::chrono::system_clock::now() - wct);
        auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(dur);
        std::cout << "@time of CPU run:\t\t\t[" << secs.count() << "] milliseconds" << std::endl;
        double dt = get_elapsed_mcycles();
        printf("@time of CPU run:\t\t\t[%.3f] million cycles\n", dt);
        minCyclesSerial = std::min(minCyclesSerial, dt);
    }

    printf("[noise serial]:\t\t[%.3f] million cycles (%d x %d image)\n", minCyclesSerial, width, height);
    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minCyclesSerial / minCyclesISPCGPU);

    // RESULT CHECK
    bool pass = true;
    double err = 0.0;
    double max_err = 0.0;

    int i = 0;
    for (; i < width * height; i++) {
        err = std::fabs(result.data[i] - gold.data[i]);
        max_err = std::max(err, max_err);
        if (err > CORRECTNESS_THRESHOLD) {
            pass = false;
            break;
        }
    }
    if (!pass) {
        std::cout << "Correctness test failed on " << i << "th value." << std::endl;
        std::cout << "Was " << result.data[i] << ", should be " << gold.data[i] << std::endl;
    } else {
        std::cout << "Passed!"
                  << " Max error:" << max_err << std::endl;
    }

    return (pass) ? 0 : 1;
}

static void usage() {
    fprintf(stderr, "usage: noise [niterations] [group threads width] [group threads height]\n");
    exit(1);
}

int main(int argc, char *argv[]) {
    int niterations = 1;
    int gx = 1, gy = 1;
    niterations = atoi(argv[1]);
    if (argc == 4) {
        gx = atoi(argv[2]);
        gy = atoi(argv[3]);
    }
    if (niterations < 1 || gx < 1 || gy < 1) {
        usage();
    }
    int success = 0;

    std::cout << "Running test with " << niterations << " iterations on " << gx << " * " << gy << " threads."
              << std::endl;
    success = run(niterations, gx, gy);

    return success;
}
