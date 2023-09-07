/*
 * Copyright (c) 2020-2023, Intel Corporation
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
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "L0_helpers.h"
#include "pipeline-dpcpp.hpp"

using namespace hostutil;

void DpcppApp::initialize() {
    if (m_initialized)
        return;
    L0InitContext(m_driver, m_device, m_context, m_module, m_command_queue, "xe_pipeline-dpcpp", false);
    L0Create_Kernel(m_device, m_context, m_module, m_command_list, m_kernel1, "stage1");
    L0Create_Kernel(m_device, m_context, m_module, m_command_list, m_kernel2, "stage3");
    m_initialized = true;
}

void DpcppApp::transformStage1(gpu_vec &in) {
    // Setup kernel arguments
    L0_SAFE_CALL(zeCommandListReset(m_command_list));
    float *data = in.data();
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel1, 0, sizeof(data), &data));
    auto data_size = in.size() * sizeof(float);
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel1, 1, sizeof(data_size), &data_size));

    // Run the ISPC kernel
    ze_group_count_t dispatchTraits = {(uint32_t)1, (uint32_t)1, 1};
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_command_list, m_kernel1, &dispatchTraits, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListClose(m_command_list));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max()));
}

void DpcppApp::transformStage2(gpu_vec &in) {
    // Create SYCL objects from native Level Zero handles
    // Thanks to this API Level Zero (ISPC) based programs
    // can share device context with SYCL programs implemented
    // using oneAPI DPC++ compiler
    auto platform = sycl::ext::oneapi::level_zero::make_platform((uintptr_t)m_driver);
    auto device = sycl::ext::oneapi::level_zero::make_device(platform, (uintptr_t)m_device);

    auto ctx = sycl::ext::oneapi::level_zero::make_context(platform.get_devices(), (uintptr_t)m_context,
                                                           /*keep ownership of m_context handler on ISPC side*/ true);
    auto q = sycl::ext::oneapi::level_zero::make_queue(
        ctx, device, (uintptr_t)m_command_queue, /* immediate command list*/ false,
        /*keep ownership of m_command_queue handler on ISPC side*/ true, sycl::property_list{});

    // Set problem space
    sycl::range<1> range{in.size()};

    // Submit a job (implemented by a lambda function) to the queue
    // This kernel works on data previously modified by ISPC kernel in Stage 1
    float *data = in.data();
    q.submit([&](cl::sycl::handler &cgh) {
         // Execute kernel in parallel instances
         cgh.parallel_for<class Stage2>(range, [=](cl::sycl::id<1> idx) {
             auto v = data[idx];
             v *= 2.0f;
             data[idx] = v;
         });
     }).wait();
}

void DpcppApp::transformStage3(gpu_vec &in) {
    // Setup kernel arguments
    // The kernel will work on memory that already is modifed in Stage 1 and Stage 2
    L0_SAFE_CALL(zeCommandListReset(m_command_list));
    float *data = in.data();
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel2, 0, sizeof(data), &data));
    auto data_size = in.size() * sizeof(float);
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel2, 1, sizeof(data_size), &data_size));

    // Run the ISPC kernel and transfer the results back from the GPU
    ze_group_count_t dispatchTraits = {(uint32_t)1, (uint32_t)1, 1};
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_command_list, m_kernel2, &dispatchTraits, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListClose(m_command_list));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max()));
}

std::vector<float> DpcppApp::transformCpu(const std::vector<float> &in) {
    std::vector<float> res(in.size());
    for (int i = 0; i < in.size(); i++) {
        res[i] = (in[i] + 1.0f) * 2.0f - 0.5f;
    }
    return res;
}

// Compare two float vectors with an Epsilon
static bool operator==(const DpcppApp::gpu_vec &l, const std::vector<float> &r) {
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

static bool operator!=(const DpcppApp::gpu_vec &l, const std::vector<float> &r) { return !(l == r); }

bool DpcppApp::run() {
    if (!m_initialized)
        throw std::runtime_error("Trying to run test on uninitialized app");

    constexpr int COUNT = 16;

    gpu_allocator<float> ga(m_device, m_context);
    gpu_vec data(COUNT, ga);
    std::generate(data.begin(), data.end(), [i = 0]() mutable { return i++; });

    // Transform the data via set of stages, forming a pipeline
    transformStage1(data);
    transformStage2(data);
    transformStage3(data);

    for (int i = 0; i < data.size(); i++) {
        std::cout << "data[" << i << "] = " << data[i] << '\n';
    }

    // Perform the same transformation on the CPU for validation purposes
    std::vector<float> data_for_val(COUNT);
    std::generate(data_for_val.begin(), data_for_val.end(), [i = 0]() mutable { return i++; });

    auto data_cpu = transformCpu(data_for_val);

    // Compare the results
    if (data != data_cpu) {
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
