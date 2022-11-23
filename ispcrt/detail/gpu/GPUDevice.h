// Copyright 2020-2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Context.h"
#include "../Device.h"
#include "../Future.h"

// std
#include <unordered_map>
#include <vector>

namespace ispcrt {
namespace gpu {

// Device discovery
uint32_t deviceCount();
ISPCRTDeviceInfo deviceInfo(uint32_t deviceIdx);

}; // gpu

struct GPUDevice : public base::Device {

    GPUDevice();
    GPUDevice(void* nativeContext, void* nativeDevice, uint32_t deviceIdx);

    ~GPUDevice();

    base::MemoryView *newMemoryView(void *appMem, size_t numBytes, const ISPCRTNewMemoryViewFlags *flags) const override;

    base::TaskQueue *newTaskQueue() const override;

    base::Module *newModule(const char *moduleFile, const ISPCRTModuleOptions &opts) const override;

    void dynamicLinkModules(base::Module **modules, const uint32_t numModules) const override;
    base::Module *staticLinkModules(base::Module **modules, const uint32_t numModules) const override;

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
    bool  m_has_context_ownership{true};
};

} // namespace ispcrt
