// Copyright 2020-2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Device.h"
#include "../Future.h"

// std
#include <unordered_map>
#include <vector>

namespace ispcrt {
namespace gpu {

uint32_t deviceCount();
ISPCRTDeviceInfo deviceInfo(uint32_t deviceIdx);

}; // gpu

struct GPUDevice : public base::Device {

    GPUDevice();
    GPUDevice(uint32_t deviceIdx);
    ~GPUDevice();

    base::MemoryView *newMemoryView(void *appMem, size_t numBytes, bool shared) const override;

    base::TaskQueue *newTaskQueue() const override;

    base::Module *newModule(const char *moduleFile, const ISPCRTModuleOptions &opts) const override;

    base::Kernel *newKernel(const base::Module &module, const char *name) const override;

    void *platformNativeHandle() const override;
    void *deviceNativeHandle() const override;
    void *contextNativeHandle() const override;

    ISPCRTAllocationType getMemAllocType(void* appMemory) const override;

  private:
    void *m_driver{nullptr};
    void *m_device{nullptr};
    void *m_context{nullptr};
    bool  m_is_mock{false};
};

} // namespace ispcrt
