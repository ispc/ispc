// Copyright 2020-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "CommandQueue.h"
#include "Kernel.h"
#include "Module.h"
#include "ModuleOptions.h"
#include "TaskQueue.h"

namespace ispcrt {
namespace base {

struct Device : public RefCounted {
    Device() = default;

    virtual ~Device() = default;

    virtual MemoryView *newMemoryView(void *appMemory, size_t numBytes,
                                      const ISPCRTNewMemoryViewFlags *flags) const = 0;

    virtual CommandQueue *newCommandQueue(uint32_t ordinal) const = 0;

    virtual TaskQueue *newTaskQueue() const = 0;

    virtual ModuleOptions *newModuleOptions() const = 0;
    virtual ModuleOptions *newModuleOptions(ISPCRTModuleType moduleType, bool libraryCompilation,
                                            uint32_t stackSize) const = 0;

    virtual Module *newModule(const char *moduleFile, const ModuleOptions &opts) const = 0;

    virtual void dynamicLinkModules(Module **modules, uint32_t numModules) const = 0;
    virtual Module *staticLinkModules(Module **modules, uint32_t numModules) const = 0;

    virtual Kernel *newKernel(const Module &module, const char *name) const = 0;

    virtual void *platformNativeHandle() const = 0;
    virtual void *deviceNativeHandle() const = 0;
    virtual void *contextNativeHandle() const = 0;

    virtual ISPCRTDeviceType getType() const = 0;

    virtual ISPCRTAllocationType getMemAllocType(void *appMemory) const = 0;
};

} // namespace base
} // namespace ispcrt
