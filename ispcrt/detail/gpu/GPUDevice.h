// Copyright 2020-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../CommandList.h"
#include "../CommandQueue.h"
#include "../Context.h"
#include "../Device.h"
#include "../Fence.h"
#include "../Future.h"

// std
#include <unordered_map>
#include <vector>

namespace ispcrt {
namespace gpu {

// Device discovery
uint32_t deviceCount();
ISPCRTDeviceInfo deviceInfo(uint32_t deviceIdx);

}; // namespace gpu

struct GPUDevice : public base::Device {

    GPUDevice();
    GPUDevice(void *nativeContext, void *nativeDevice, uint32_t deviceIdx);

    ~GPUDevice();

    base::MemoryView *newMemoryView(void *appMem, size_t numBytes,
                                    const ISPCRTNewMemoryViewFlags *flags) const override;

    base::CommandQueue *newCommandQueue(uint32_t ordinal) const override;

    base::TaskQueue *newTaskQueue() const override;

    base::ModuleOptions *newModuleOptions() const override;
    base::ModuleOptions *newModuleOptions(ISPCRTModuleType moduleType, bool libraryCompilation,
                                          uint32_t stackSize) const override;

    base::Module *newModule(const char *moduleFile, const base::ModuleOptions &opts) const override;

    void dynamicLinkModules(base::Module **modules, const uint32_t numModules) const override;
    base::Module *staticLinkModules(base::Module **modules, const uint32_t numModules) const override;

    base::Kernel *newKernel(const base::Module &module, const char *name) const override;

    void *platformNativeHandle() const override;
    void *deviceNativeHandle() const override;
    void *contextNativeHandle() const override;

    ISPCRTDeviceType getType() const override;

    ISPCRTAllocationType getMemAllocType(void *appMemory) const override;

  private:
    void *m_driver{nullptr};
    void *m_device{nullptr};
    void *m_context{nullptr};
    bool m_is_mock{false};
    bool m_has_context_ownership{true};
};

} // namespace ispcrt

// Expose API of GPU device solib for dlsym.
extern "C" {
ispcrt::base::Device *load_gpu_device();
ispcrt::base::Device *load_gpu_device_ctx(void *ctx, void *dev, uint32_t idx);
uint32_t gpu_device_count();
ISPCRTDeviceInfo gpu_device_info(uint32_t idx);
}
