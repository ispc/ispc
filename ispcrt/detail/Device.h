// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "Kernel.h"
#include "Module.h"
#include "TaskQueue.h"

namespace ispcrt {

struct Device : public RefCounted {
    Device() = default;
    virtual ~Device() = default;

    virtual MemoryView *newMemoryView(void *appMemory, size_t numBytes) const = 0;

    virtual TaskQueue *newTaskQueue() const = 0;

    virtual Module *newModule(const char *moduleFile) const = 0;

    virtual Kernel *newKernel(const Module &module, const char *name) const = 0;
};

} // namespace ispcrt
