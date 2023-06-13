// Copyright 2020-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../CommandQueue.h"
#include "../Device.h"
#include "../Future.h"
#include "../ModuleOptions.h"

namespace ispcrt {

namespace cpu {

uint32_t deviceCount();
ISPCRTDeviceInfo deviceInfo(uint32_t deviceIdx);

}; // namespace cpu

struct CPUDevice : public base::Device {
    CPUDevice() = default;

    base::MemoryView *newMemoryView(void *appMem, size_t numBytes,
                                    const ISPCRTNewMemoryViewFlags *flags) const override;

    base::CommandQueue *newCommandQueue(uint32_t ordinal) const override;

    base::TaskQueue *newTaskQueue() const override;

    base::ModuleOptions *newModuleOptions() const override;
    base::ModuleOptions *newModuleOptions(ISPCRTModuleType moduleType, bool libraryCompilation,
                                          uint32_t stackSize) const override;

    base::Module *newModule(const char *moduleFile, const base::ModuleOptions &moduleOpts) const override;

    void dynamicLinkModules(base::Module **modules, const uint32_t numModules) const override;
    base::Module *staticLinkModules(base::Module **modules, const uint32_t numModules) const override;

    base::Kernel *newKernel(const base::Module &module, const char *name) const override;

    void *platformNativeHandle() const override;
    void *deviceNativeHandle() const override;
    void *contextNativeHandle() const override;

    ISPCRTDeviceType getType() const override;

    ISPCRTAllocationType getMemAllocType(void *appMemory) const override;
};

} // namespace ispcrt

// Expose API of CPU device solib for dlsym.
extern "C" {
ispcrt::base::Device *load_cpu_device();
uint32_t cpu_device_count();
ISPCRTDeviceInfo cpu_device_info(uint32_t idx);
}
