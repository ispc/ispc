// Copyright 2020-2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "Kernel.h"
#include "Module.h"
#include "TaskQueue.h"

namespace ispcrt {
namespace base {

struct Device : public RefCounted {
    Device() = default;

    virtual ~Device() = default;

    virtual MemoryView *newMemoryView(void *appMemory, size_t numBytes, bool shared) const = 0;

    virtual TaskQueue *newTaskQueue() const = 0;

    virtual Module *newModule(const char *moduleFile, const ISPCRTModuleOptions &opts) const = 0;

    virtual Kernel *newKernel(const Module &module, const char *name) const = 0;

    virtual void *platformNativeHandle() const = 0;
    virtual void *deviceNativeHandle() const = 0;
    virtual void *contextNativeHandle() const = 0;

    virtual ISPCRTAllocationType getMemAllocType(void* appMemory) const = 0;
};

} // namespace base
} // namespace ispcrt
