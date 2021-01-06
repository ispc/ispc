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

#include <level_zero/ze_api.h>
#include "L0_helpers.h"

template <typename T>
struct gpu_allocator {
    using value_type      = T;

    gpu_allocator() = delete;
    gpu_allocator(ze_device_handle_t device, ze_context_handle_t context) : m_device {device}, m_context {context} {}
    gpu_allocator(const gpu_allocator&) = default;
    ~gpu_allocator() = default;
    gpu_allocator& operator=(const gpu_allocator&) = delete;

    T* allocate(const size_t n) const;
    void deallocate(T* const p, const size_t n) const;
private:
    ze_device_handle_t  m_device{nullptr};
    ze_context_handle_t m_context{nullptr};
};

template <typename T>
inline T *gpu_allocator<T>::allocate(const size_t n) const {
    ze_device_mem_alloc_desc_t device_alloc_desc = {};
    ze_host_mem_alloc_desc_t host_alloc_desc = {};

    // Allocate a memory chunks that can be shared between the host and the device
    void* ptr = nullptr;

    auto status = zeMemAllocShared(m_context, &device_alloc_desc, &host_alloc_desc, n, alignof(T), m_device, (void**)&ptr);
    if (status != 0) {
        throw std::runtime_error("gpu_allocator<T>::allocate() - Level Zero error");
    }
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }

    return static_cast<T*>(ptr);
}

template <typename T>
inline void gpu_allocator<T>::deallocate(T* const p, const size_t) const {
    zeMemFree(m_context, p);
}

class DpcppApp {
  public:
    DpcppApp() = default;

    void initialize();
    bool run();
    void cleanup();

    using gpu_vec = std::vector<float, gpu_allocator<float>>;

    // Transformation passes
    void transformStage1(gpu_vec& in); // ISPC
    void transformStage2(gpu_vec& in); // DPC++
    void transformStage3(gpu_vec& in); // ISPC

    // Validation is done on the CPU
    std::vector<float> transformCpu(const std::vector<float>& in);

  private:
    bool m_initialized{false};
    ze_driver_handle_t m_driver{nullptr};
    ze_device_handle_t m_device{nullptr};
    ze_module_handle_t m_module{nullptr};
    ze_kernel_handle_t m_kernel1{nullptr};
    ze_kernel_handle_t m_kernel2{nullptr};
    ze_context_handle_t m_context{nullptr};
    ze_command_list_handle_t m_command_list{nullptr};
    ze_command_queue_handle_t m_command_queue{nullptr};
};
