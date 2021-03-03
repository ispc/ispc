// Copyright 2020-2021 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Future.h"
#include "Kernel.h"
#include "MemoryView.h"

namespace ispcrt {
namespace base {

struct TaskQueue : public RefCounted {
    TaskQueue() = default;
    virtual ~TaskQueue() = default;

    virtual void barrier() = 0;

    virtual void copyToHost(base::MemoryView &mv) = 0;
    virtual void copyToDevice(base::MemoryView &mv) = 0;

    virtual base::Future *launch(Kernel &k, base::MemoryView *params, size_t dim0, size_t dim1, size_t dim2) = 0;

    virtual void submit() = 0;
    virtual void sync() = 0;

    virtual void* taskQueueNativeHandle() const = 0;
};

} // namespace base
} // namespace ispcrt
