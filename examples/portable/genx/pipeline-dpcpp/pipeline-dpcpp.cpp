/*
 * Copyright (c) 2020, Intel Corporation
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

#include <iostream>
#include <vector>

// Level Zero headers
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

// SYCL and interoperability headers
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>

#include "L0_helpers.h"
#include "pipeline-dpcpp.hpp"

using namespace hostutil;

void DpcppApp::initialize() {
    if (m_initialized)
        return;
    L0InitContext(m_driver, m_device, m_context, m_module, m_command_queue, "genx_pipeline-dpcpp", false);
    L0Create_Kernel(m_device, m_context, m_module, m_command_list, m_kernel1, "stage1");
    L0Create_Kernel(m_device, m_context, m_module, m_command_list, m_kernel2, "stage3");
    m_initialized = true;
}

void DpcppApp::transformStage1(const std::vector<float>& in) {
    m_count = in.size();
    const auto data_size = sizeof(float) * m_count;

    ze_device_mem_alloc_desc_t device_alloc_desc = {};
    ze_host_mem_alloc_desc_t host_alloc_desc = {};

    // Allocate a memory chunks that can be shared between the host and the device
    L0_SAFE_CALL(zeMemAllocShared(m_context, &device_alloc_desc, &host_alloc_desc, data_size, alignof(float),
                                  m_device, (void**)&m_shared_data));

    std::memcpy(m_shared_data, in.data(), data_size);

    // Setup kernel arguments
    L0_SAFE_CALL(zeCommandListReset(m_command_list));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel1, 0, sizeof(m_shared_data), &m_shared_data));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel1, 1, sizeof(m_count), &m_count));
    // Let the runtime know that kernel will access the shared memory
    ze_kernel_indirect_access_flags_t mem_flags = ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
    L0_SAFE_CALL(zeKernelSetIndirectAccess(m_kernel1, mem_flags));

    // Run the ISPC kernel
    ze_group_count_t dispatchTraits = {(uint32_t)1, (uint32_t)1, 1};
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_command_list, m_kernel1, &dispatchTraits, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListClose(m_command_list));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max()));
}

void DpcppApp::transformStage2() {
    // Create SYCL objects from native Level Zero handles
    // Thanks to this API Level Zero (ISPC) based programs
    // can share device context with SYCL programs implemented
    // using oneAPI DPC++ compiler
    auto platform = sycl::level_zero::make<cl::sycl::platform>(m_driver);
    auto device   = sycl::level_zero::make<cl::sycl::device>(platform, m_device);
    auto ctx      = sycl::level_zero::make<cl::sycl::context>(platform.get_devices(), m_context);
    auto q        = sycl::level_zero::make<cl::sycl::queue>(ctx, m_command_queue);

    // Set problem space
    sycl::range<1> range { m_count };

    // Submit a job (implemented by a lambda function) to the queue
    auto mem = m_shared_data; // cannot dereference *this on the device, so we need a copy
    // This kernel works on data previously modified by ISPC kernel in Stage 1
    q.submit([&](cl::sycl::handler &cgh) {
        // Execute kernel in parallel instances
        cgh.parallel_for<class Stage2>(range, [=](cl::sycl::id<1> idx) {
            auto v = mem[idx];
            v *= 2.0f;
            mem[idx] = v;
        });
    }).wait();
}

std::vector<float> DpcppApp::transformStage3() {
    std::vector<float> out(m_count, 0.0f);

    // Setup kernel arguments
    // The kernel will work on memory that already is modifed in Stage 1 and Stage 2
    L0_SAFE_CALL(zeCommandListReset(m_command_list));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel2, 0, sizeof(m_shared_data), &m_shared_data));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel2, 1, sizeof(m_count), &m_count));

    // Run the ISPC kernel and transfer the results back from the GPU
    ze_group_count_t dispatchTraits = {(uint32_t)1, (uint32_t)1, 1};
    ze_kernel_indirect_access_flags_t mem_flags = ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
    L0_SAFE_CALL(zeKernelSetIndirectAccess(m_kernel2, mem_flags));
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_command_list, m_kernel2, &dispatchTraits, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListClose(m_command_list));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max()));

    // Read back the memory from the device
    std::memcpy(out.data(), m_shared_data, sizeof(float) * m_count);

    // Perform cleanup of the shared memory space
    L0_SAFE_CALL(zeMemFree(m_context, m_shared_data));

    return out;
}

std::vector<float> DpcppApp::transformCpu(const std::vector<float>& in) {
    std::vector<float> res(in.size());
    for (int i = 0; i < in.size(); i++) {
        res[i] = (in[i] + 1.0f) * 2.0f - 0.5f;
    }
    return res;
}

// Compare two float vectors with an Epsilon
static bool operator==(const std::vector<float>& l, const std::vector<float>& r) {
    constexpr float EPSILON = 0.01f;
    if (l.size() != r.size())
        return false;
    for (unsigned int i = 0; i < l.size(); i++) {
        if ((float)fabs(r[i] - l[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

bool DpcppApp::run() {
    if (!m_initialized)
        throw std::runtime_error("Trying to run test on uninitialized app");

    constexpr int COUNT = 16;
    std::vector<float> vin(COUNT);

    std::generate(vin.begin(), vin.end(), [i = 0]() mutable { return i++; });
    // Transform the data via set of stages, forming a pipeline
    transformStage1(vin);
    transformStage2();
    auto vout = transformStage3();

    for (int i = 0; i < vout.size(); i++) {
        std::cout << "out[" << i << "] = " << vout[i] << '\n';
    }

    // Perform the same transformation on the CPU for validation purposes
    auto vout_cpu = transformCpu(vin);

    // Compare the results
    if (vout != vout_cpu) {
        std::cout << "Validation failed!\n";
        return false;
    }

    std::cout << "Validation passed!\n";
    return true;
}

void DpcppApp::cleanup() {
    L0_SAFE_CALL(zeKernelDestroy(m_kernel1));
    L0_SAFE_CALL(zeKernelDestroy(m_kernel2));
    L0_SAFE_CALL(zeCommandListDestroy(m_command_list));
    L0DestroyContext(m_driver, m_device, m_context, m_module, m_command_queue);
    m_initialized = false;
}

int main() {
    DpcppApp app;

    app.initialize();
    bool pass = app.run();
    app.cleanup();
    return pass ? 0 : -1;
}
