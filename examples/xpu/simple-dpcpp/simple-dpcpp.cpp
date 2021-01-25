/*
 * Copyright (c) 2021, Intel Corporation
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
#include <iomanip>
#include <vector>

// Level Zero headers
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

// SYCL and interoperability headers
#include <CL/sycl.hpp>
#include <CL/sycl/backend/level_zero.hpp>

#include "L0_helpers.h"
#include "simple-dpcpp.hpp"

using namespace hostutil;

DpcppApp::DpcppApp() {
    m_device = ispcrt::Device(ISPCRT_DEVICE_TYPE_GPU);
    m_module = ispcrt::Module(m_device, "genx_simple-dpcpp");
    m_kernel = ispcrt::Kernel(m_device, m_module, "simple_ispc");
    m_queue  = ispcrt::TaskQueue(m_device);
    initialized = true;
}

std::vector<float> DpcppApp::transformIspc(std::vector<float>& in) {
    const auto count = in.size();
    std::vector<float> out(count, 0.0f);

    // Setup input array
    ispcrt::Array<float> in_dev(m_device, in);

    // Setup output array
    ispcrt::Array<float> out_dev(m_device, out);

    // Setup parameters structure
    struct Parameters {
        float *in;
        float *out;
        int    count;
    };

    Parameters p;

    p.in = in_dev.devicePtr();
    p.out = out_dev.devicePtr();
    p.count = count;

    auto p_dev = ispcrt::Array<Parameters>(m_device, p);

    // ispcrt::Array objects which used as inputs for ISPC kernel should be
    // explicitly copied to device from host
    m_queue.copyToDevice(p_dev);
    m_queue.copyToDevice(in_dev);

    // Make sure that input arrays were copied
    m_queue.barrier();

    // Launch the kernel on the device using 1 thread
    m_queue.launch(m_kernel, p_dev, 1);

    // Make sure that execution completed
    m_queue.barrier();

    // ispcrt::Array objects which used as outputs of ISPC kernel should be
    // explicitly copied to host from device
    m_queue.copyToHost(out_dev);

    // Make sure that input arrays were copied
    m_queue.barrier();

    // Execute queue and sync
    m_queue.sync();

    return out;
}

std::vector<float> DpcppApp::transformDpcpp(const std::vector<float>& in) {
    const auto count = in.size();
    std::vector<float> out(count, 0.0f);

    // Create SYCL objects from native (Level Zero) handles
    // Thanks to this API ISPCRT-based based programs
    // can share device context with SYCL programs implemented
    // using oneAPI DPC++ compiler
    auto nativePlatform = static_cast<ze_driver_handle_t>(m_device.nativePlatformHandle());
    auto nativeDevice   = static_cast<ze_device_handle_t>(m_device.nativeDeviceHandle());
    auto nativeContext  = static_cast<ze_context_handle_t>(m_device.nativeContextHandle());
    auto nativeQueue    = static_cast<ze_command_queue_handle_t>(m_queue.nativeTaskQueueHandle());

    auto platform = sycl::level_zero::make<cl::sycl::platform>(nativePlatform);
    auto device   = sycl::level_zero::make<cl::sycl::device>(platform, nativeDevice);
    auto ctx      = sycl::level_zero::make<cl::sycl::context>(platform.get_devices(), nativeContext);
    auto q        = sycl::level_zero::make<cl::sycl::queue>(ctx, nativeQueue);

    // Set problem space
    sycl::range<1> range { count };
    // Allocate buffers used for communication with the device
    sycl::buffer<float, 1> in_buffer(in.data(), range);
    sycl::buffer<float, 1> out_buffer(out.data(), range);

    // Submit a job (implemented by a lambda function) to the queue
    q.submit([&](cl::sycl::handler &cgh) {
        // Accessors are used to access buffers on the device and on the host
        auto in_access  = in_buffer.get_access<cl::sycl::access::mode::read>(cgh);
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
    for (int i = 0; i < out_host_access.get_count(); i++) {
        res[i] = out_host_access[i];
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

    std::cout << "           ISPC   DPCPP\n";
    constexpr unsigned COLW = 6;
    constexpr unsigned PREC = 3;
    for (int i = 0; i < COUNT; i++) {
        std::cout << "out[" << std::setw(2) << i << "] = "
                  << std::fixed << std::setw(COLW) << std::setprecision(PREC) << vout_ispc[i]
                  << std::fixed << std::setw(COLW) << std::setprecision(PREC) << vout_dpcpp[i] << '\n';
    }

    // Compare the results
    if (vout_dpcpp != vout_ispc) {
        std::cout << "Validation failed!\n";
        return false;
    }

    std::cout << "Validation passed!\n";
    return true;
}

int main() {
    DpcppApp app;
    bool pass = app.run();
    return pass ? 0 : -1;
}
