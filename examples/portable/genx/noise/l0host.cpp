#include <iostream>
#include <level_zero/ze_api.h>
#include <limits>
#include <cmath>
#include "L0_helpers.h"

// TODO: move out..
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

    for (int i = 0; i < SZ; i++)
        out.data[i] = -1;

    L0InitContext(hDriver, hDevice, hModule, hCommandQueue);

    ze_command_list_handle_t hCommandList;
    ze_kernel_handle_t hKernel;

    L0Create_Kernel(hDevice, hModule, hCommandList, hKernel, "noise_ispc");

    // PARAMS
    const unsigned int height = 768;
    const unsigned int width = 768;

    const float x0 = -10;
    const float y0 = -10;
    const float x1 = 10;
    const float y1 = 10;

    void *buf_ref = out.data;
    L0_SAFE_CALL(zeDriverAllocSharedMem(hDriver, hDevice, ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT, 0,
                                        ZE_HOST_MEM_ALLOC_FLAG_DEFAULT, SZ * sizeof(float), SZ * sizeof(float),
                                        &buf_ref));

    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 0, sizeof(float), &x0));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 1, sizeof(float), &y0));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 2, sizeof(float), &x1));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 3, sizeof(float), &y1));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 4, sizeof(int), &width));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 5, sizeof(int), &height));
    L0_SAFE_CALL(zeKernelSetArgumentValue(hKernel, 6, sizeof(buf_ref), &buf_ref));


    // EXECUTION
    // set group size
    // FIXME
    uint32_t groupSpaceWidth = gx;
    uint32_t groupSpaceHeight = gy;
    
    uint32_t group_size = groupSpaceWidth * groupSpaceHeight;
    L0_SAFE_CALL(zeKernelSetGroupSize(hKernel, /*x*/ groupSpaceWidth, /*y*/ groupSpaceHeight, /*z*/ 1));

    // set grid size
    ze_thread_group_dimensions_t dispatchTraits = {1, 1, 1};

    // launch
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(hCommandList, hKernel, &dispatchTraits, nullptr, 0, nullptr));

    L0_SAFE_CALL(zeCommandListAppendBarrier(hCommandList, nullptr, 0, nullptr));

    // copy result to host
    void *res = result.data;
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(hCommandList, res, buf_ref, SZ*sizeof(float), nullptr));
    // dispatch & wait
    L0_SAFE_CALL(zeCommandListClose(hCommandList));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(hCommandQueue, std::numeric_limits<uint32_t>::max()));

    // TODO: timeit!
    /*auto timings = execute(device, kernel, gx, gy, niter, false, TIMEOUT);
    timings.print(niter);*/

    //void *res = out.data;
    L0_SAFE_CALL(zeDriverFreeMem(hDriver, buf_ref));

    // RESULT CHECK
    bool pass = true;
    if (niter == 1) {
        noise_serial(x0, y0, x1, y1, width, height, gold.data);
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
    }

    return (pass) ? 0 : 1;
}

int main(int argc, char *argv[]) {
    int niterations = 1;
    int gx = 1, gy = 1;
    niterations = atoi(argv[1]);
    if (argc == 4) {
        gx = atoi(argv[2]);
        gy = atoi(argv[3]);
    }

    int success = 0;

    std::cout << "Running test with " << niterations << " iterations on " << gx << " * " << gy << " threads."
              << std::endl;
    success = run(niterations, gx, gy);

    return success;
}

