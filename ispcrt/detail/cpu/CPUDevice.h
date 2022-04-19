// Copyright 2020-2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Device.h"
#include "../Future.h"

namespace ispcrt {

namespace cpu {

uint32_t deviceCount();
ISPCRTDeviceInfo deviceInfo(uint32_t deviceIdx);

}; // cpu

struct CPUDevice : public base::Device {
    CPUDevice() = default;

    base::MemoryView *newMemoryView(void *appMem, size_t numBytes, bool shared) const override;

    base::TaskQueue *newTaskQueue() const override;

    base::Module *newModule(const char *moduleFile, const ISPCRTModuleOptions &moduleOpts) const override;

    base::Kernel *newKernel(const base::Module &module, const char *name) const override;

    void *platformNativeHandle() const override;
    void *deviceNativeHandle() const override;
    void *contextNativeHandle() const override;

    ISPCRTAllocationType getMemAllocType(void* appMemory) const override;
};

} // namespace ispcrt
