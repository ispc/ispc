/*
 * Copyright (c) 2020-2021, Intel Corporation
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
#include <sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include "L0_helpers.h"
#include "simple-dpcpp-l0.hpp"

using namespace hostutil;

void DpcppApp::initialize() {
    if (initialized)
        return;
    L0InitContext(m_driver, m_device, m_context, m_module, m_command_queue, "xe_simple-dpcpp-l0", false);
    L0Create_Kernel(m_device, m_context, m_module, m_command_list, m_kernel, "simple_ispc");
    initialized = true;
}

std::vector<float> DpcppApp::transformIspc(const std::vector<float> &in) {
    const auto count = in.size();
    std::vector<float> out(count, 0.0f);

    void *in_dev = nullptr;
    void *out_dev = nullptr;
    void *params_dev = nullptr;

    struct Parameters {
        float *in;
        float *out;
        int count;
    };

    Parameters params;

    ze_device_mem_alloc_desc_t alloc_desc = {};

    // Allocate memory on the device
    L0_SAFE_CALL(zeMemAllocDevice(m_context, &alloc_desc, count * sizeof(float), 0, m_device, &in_dev));
    L0_SAFE_CALL(zeMemAllocDevice(m_context, &alloc_desc, count * sizeof(float), 0, m_device, &out_dev));
    L0_SAFE_CALL(zeMemAllocDevice(m_context, &alloc_desc, sizeof(Parameters), 0, m_device, &params_dev));

    params.in = reinterpret_cast<float *>(in_dev);
    params.out = reinterpret_cast<float *>(out_dev);
    params.count = count;

    // Enqueue memory transfers, setup kernel arguments and prepare synchronization
    L0_SAFE_CALL(zeCommandListReset(m_command_list));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_command_list, in_dev, in.data(), in.size() * sizeof(float), nullptr, 0,
                                               nullptr));
    L0_SAFE_CALL(
        zeCommandListAppendMemoryCopy(m_command_list, params_dev, &params, sizeof(Parameters), nullptr, 0, nullptr));
    L0_SAFE_CALL(zeKernelSetArgumentValue(m_kernel, 0, sizeof(&params), &params_dev));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));

    // Run the ISPC kernel and transfer the results back from the GPU
    ze_group_count_t dispatchTraits = {(uint32_t)1, (uint32_t)1, 1};
    L0_SAFE_CALL(zeCommandListAppendLaunchKernel(m_command_list, m_kernel, &dispatchTraits, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendMemoryCopy(m_command_list, out.data(), out_dev, out.size() * sizeof(float), nullptr,
                                               0, nullptr));
    L0_SAFE_CALL(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    L0_SAFE_CALL(zeCommandListClose(m_command_list));
    L0_SAFE_CALL(zeCommandQueueExecuteCommandLists(m_command_queue, 1, &m_command_list, nullptr));
    L0_SAFE_CALL(zeCommandQueueSynchronize(m_command_queue, std::numeric_limits<uint64_t>::max()));

    // Perform cleanup
    L0_SAFE_CALL(zeMemFree(m_context, in_dev));
    L0_SAFE_CALL(zeMemFree(m_context, out_dev));
    L0_SAFE_CALL(zeMemFree(m_context, params_dev));

    return out;
}

std::vector<float> DpcppApp::transformDpcpp(const std::vector<float> &in) {
    const auto count = in.size();
    std::vector<float> out(count, 0.0f);

    // Create SYCL objects from native Level Zero handles
    // Thanks to this API Level Zero (ISPC) based programs
    // can share device context with SYCL programs implemented
    // using oneAPI DPC++ compiler
    auto platform = sycl::ext::oneapi::level_zero::make_platform((uintptr_t)m_driver);
    auto device = sycl::ext::oneapi::level_zero::make_device(platform, (uintptr_t)m_device);

    auto ctx = sycl::ext::oneapi::level_zero::make_context(platform.get_devices(), (uintptr_t)m_context,
                                                           /*keep ownership of m_context handler on ISPC side*/ true);
    auto q = sycl::ext::oneapi::level_zero::make_queue(ctx, (uintptr_t)m_command_queue,
                                                       /*keep ownership of m_command_queue handler on ISPC side*/ true);

    // Set problem space
    sycl::range<1> range{count};
    // Allocate buffers used for communication with the device
    sycl::buffer<float, 1> in_buffer(in.data(), range);
    sycl::buffer<float, 1> out_buffer(out.data(), range);

    // Submit a job (implemented by a lambda function) to the queue
    q.submit([&](cl::sycl::handler &cgh) {
        // Accessors are used to access buffers on the device and on the host
        auto in_access = in_buffer.get_access<cl::sycl::access::mode::read>(cgh);
        auto out_access = out_buffer.get_access<cl::sycl::access::mode::write>(cgh);

        // Execute kernel in parallel instances
        cgh.parallel_for<class DpcppSimple>(range, [=](cl::sycl::id<1> idx) {
            auto v = in_access[idx];
            // This is the same computation as in ISPC kernel so we can compare the results
            if (v < 3.)
                v = v * v;
            else
                v = cl::sycl::sqrt(v);
            out_access[idx] = v;
        });
    });

    // Use accessor to transfer data from the device
    std::vector<float> res(count);
    const auto out_host_access = out_buffer.get_access<cl::sycl::access::mode::read>();
    for (int i = 0; i < out_host_access.size(); i++) {
        res[i] = out_host_access[i];
    }
    return res;
}

// Compare two float vectors with an Epsilon
static bool operator==(const std::vector<float> &l, const std::vector<float> &r) {
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
    if (!initialized)
        throw std::runtime_error("Trying to run test on uninitialized app");

    constexpr int COUNT = 16;
    std::vector<float> vin(COUNT);

    // Make some changes (the same as in the ISPC Simple example)
    // on a buffer using ISPC kernel
    std::generate(vin.begin(), vin.end(), [i = 0]() mutable { return i++; });
    auto vout_ispc = transformIspc(vin);

    // Perform the same action using features of oneAPI DPC++ Compiler
    std::generate(vin.begin(), vin.end(), [i = 0]() mutable { return i++; });
    auto vout_dpcpp = transformDpcpp(vin);

    for (int i = 0; i < vout_dpcpp.size(); i++) {
        std::cout << "out[" << i << "] = " << vout_dpcpp[i] << '\n';
    }

    // Compare the results
    if (vout_dpcpp != vout_ispc) {
        std::cout << "Validation failed!\n";
        return false;
    }

    std::cout << "Validation passed!\n";
    return true;
}

void DpcppApp::cleanup() {
    L0Destroy_Kernel(m_command_list, m_kernel);
    L0DestroyContext(m_driver, m_device, m_context, m_module, m_command_queue);
    initialized = false;
}

int main() {
    DpcppApp app;

    app.initialize();
    bool pass = app.run();
    app.cleanup();
    return pass ? 0 : -1;
}
