/*
 * Copyright (c) 2019-2021, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include "L0_helpers.h"
#include "Matrix.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

#include "sgemm.hpp"

using namespace hostutil;

void SGEMMApp::initialize() {
    if (initialized)
        return;

    auto useZebin = getenv("ISPC_EXAMPLES_USE_ZEBIN") != nullptr;
    if (!useZebin) {
        useZebin = getenv("ISPCRT_USE_ZEBIN") != nullptr;
    }

#ifdef CMKERNEL
    L0InitContext(m_driver, m_device, m_context, m_module, m_command_queue, "naive_sgemm_cm_mt.spv");
#else
    L0InitContext(m_driver, m_device, m_context, m_module, m_command_queue, "xe_sgemm", useZebin);
#endif

    // Get device timestamp resolution - needed for time measurments
    ze_device_properties_t device_properties;
    L0_SAFE_CALL(zeDeviceGetProperties(m_device, &device_properties));
    m_timestamp_freq = device_properties.timerResolution;

#ifdef CMKERNEL
    if (m_verbose)
        std::cout << "Running CM kernel\n";
    L0Create_Kernel(m_device, m_context, m_module, m_command_list, m_kernel, "sgemm_kernel");
#else
    if (m_verbose)
        std::cout << "Running ISPC kernel\n";
    L0Create_Kernel(m_device, m_context, m_module, m_command_list, m_kernel, "SGEMM_naive_task");
#endif

    L0Create_EventPool(m_device, m_context, 1, m_pool);

    // Create event used to measure kernel execution time
    ze_event_desc_t eventDesc = {};
    eventDesc.index = 0;
    eventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

    L0_SAFE_CALL(zeEventCreate(m_pool, &eventDesc, &m_event));

    initialized = true;
}

void SGEMMApp::run(SGEMMApp::RunResult &result, int m, int niter, int gx, int gy, bool validate) {
    if (!initialized)
        throw std::runtime_error("Trying to run test on uninitialized app");

    std::cout.setf(std::ios::unitbuf);

    storage_type_t st = RowMajor;
    float alpha = +1.0, beta = +0.0;

    int n = m, k = m;

    // Initialization
    m = (m / TILE_m) * TILE_m;
    n = k = m;

    int lda = ((k + 15) & ~15);
    int ldb = ((n + 15) & ~15);
    int ldc = ldb;

    const int lda_tmp = (((GEMM_BLOCK / TILE_m) * TILE_m) + 15) & ~15;
    const int ldb_tmp = lda_tmp;
    const int ldc_tmp = lda_tmp;
    if (m_verbose) {
        printf("SGEMM: C(%d, %d) = %.2f * C(%d, %d) + %.2f A(%d, %d) * B(%d, %d)\n", m, n, beta, m, n, alpha, m, k, k,
               n);
        printf("Thread-group setting: %d x %d \n", gx, gy);
    }
    // Allocate matrices
    Matrix A(m, k, lda, NULL, true, "A", st);
    Matrix B(k, n, ldb, NULL, true, "B", st);
    Matrix C(m, n, ldc, NULL, false, "C", st);
    Matrix C_gold(C, "C_gold");

    if (validate) {
        if (m_verbose) {
            printf("** validation run, only one iteration **\n");
            printf("** For performance run, add cmd-args: Sgemm 2048 1000 ** \n");
            printf("Compute gold result\n");
        }
        // Compute gold result
        sgemmNxN(m, n, k, alpha, &A(0, 0), A.l_dim(), &B(0, 0), B.l_dim(), beta, &C_gold(0, 0), C_gold.l_dim(), st);
    } else {
        if (m_verbose)
            printf("CPU result not computed: Make #iterations=1 to compute CPU result\n");
    }

    void *a_buf = nullptr;
    void *b_buf = nullptr;
    void *c_buf = nullptr;

    int mtA_size = A.l_dim() * m;
    int mtB_size = B.l_dim() * B.n_row();
    int mtC_size = C.l_dim() * m;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            C(i, j) = -1;

    ze_device_mem_alloc_desc_t alloc_desc = {};

    L0_SAFE_CALL(zeMemAllocDevice(m_context, &alloc_desc, mtA_size * sizeof(float), 0, m_device, &a_buf));
    L0_SAFE_CALL(zeMemAllocDevice(m_context, &alloc_desc, mtB_size * sizeof(float), 0, m_device, &b_buf));
    L0_SAFE_CALL(zeMemAllocDevice(m_context, &alloc_desc, mtC_size * sizeof(float), 0, m_device, &c_buf));

    L0_SAFE_CALL(zeCommandListReset(m_command_list));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(m_command_list, a_buf, &A(0, 0), mtA_size * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(m_command_list, b_buf, &B(0, 0), mtB_size * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(m_command_list, c_buf, &C(0, 0), mtC_size * sizeof(float), nullptr, 0, nullptr));

    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel, 0, sizeof(a_buf), &a_buf));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel, 1, sizeof(b_buf), &b_buf));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel, 2, sizeof(c_buf), &c_buf));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel, 3, sizeof(int), &m));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel, 4, sizeof(int), &n));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel, 5, sizeof(int), &k));

    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));

    // EXECUTION
    // set group size
    uint32_t groupSpaceWidth = 1;
    uint32_t groupSpaceHeight = 1;

    uint32_t group_size = groupSpaceWidth * groupSpaceHeight;
    L0_SAFE_CALL(zeKernelSetGroupSize(m_kernel, /*x*/ groupSpaceWidth, /*y*/ groupSpaceHeight, /*z*/ 1));

    // set grid size
    ze_group_count_t dispatchTraits = {(uint32_t)gx, (uint32_t)gy, 1};
    if (m_verbose) {
        std::cout << "Set dispatchTraits.x=" << dispatchTraits.groupCountX
                  << ", dispatchTraits.y=" << dispatchTraits.groupCountY << std::endl;
    }

    std::chrono::duration<uint64_t, std::nano> gpu_duration(0);
    auto tot_wct = std::chrono::system_clock::now();

    for (int i = 0; i < niter; ++i) {
        L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_command_list, m_kernel, &dispatchTraits, m_event, 0, nullptr));
        L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
        L0_SAFE_CALL(zeCommandListClose(m_command_list));
        L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr));
        L0_SAFE_CALL(zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max()));
        L0_SAFE_CALL(zeCommandListReset(m_command_list));
        // get time
        ze_kernel_timestamp_result_t tsResult;
        L0_SAFE_CALL(zeEventQueryKernelTimestamp(m_event, &tsResult));

        std::chrono::duration<uint64_t, std::nano> event_dur(
            m_timestamp_freq * (tsResult.context.kernelEnd - tsResult.context.kernelStart));
        gpu_duration += event_dur;
        L0_SAFE_CALL(zeEventHostReset(m_event));
    }
    auto tot_dur = std::chrono::system_clock::now() - tot_wct;
    auto tot_nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(tot_dur);
    auto gpu_nsecs = std::chrono::nanoseconds(gpu_duration);

    if (m_verbose) {
        using double_mili = std::chrono::duration<double, std::chrono::milliseconds::period>;
        printf("@Average execution time is:\t\t\t[%f] milliseconds\n", double_mili(tot_nsecs).count() / niter);
        printf("@Average GPU time is:\t\t\t[%f] milliseconds\n", double_mili(gpu_nsecs).count() / niter);
    }

    // copy result to host
    // L0_SAFE_CALL(zeCommandListReset(m_command_list));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(m_command_list, &C(0, 0), c_buf, mtC_size * sizeof(float), nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListClose(m_command_list));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max()));
    // Result check
    bool pass = true;
    if (validate) {
        if (C != C_gold) {
            pass = false;
            if (m_verbose)
                printf("Result is NOT correct!\n");
        }
    } else {
        if (m_verbose)
            printf("Result not checked - make #iterations=1 to check result!\n");
    }

    L0_SAFE_CALL(zeMemFree(m_context, a_buf));
    L0_SAFE_CALL(zeMemFree(m_context, b_buf));
    L0_SAFE_CALL(zeMemFree(m_context, c_buf));

    result.valid = pass;
    result.cpuTime = tot_nsecs.count();
    result.gpuTime = gpu_nsecs.count();
}

void SGEMMApp::cleanup() {
    L0_SAFE_CALL(zeEventDestroy(m_event));

    L0Destroy_EventPool(m_pool);
    L0Destroy_Kernel(m_command_list, m_kernel);
    L0DestroyContext(m_driver, m_device, m_context, m_module, m_command_queue);

    initialized = false;
}
