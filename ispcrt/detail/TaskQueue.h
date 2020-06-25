// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "Future.h"
#include "Kernel.h"
#include "MemoryView.h"

namespace ispcrt {

struct TaskQueue : public RefCounted {
    TaskQueue() = default;
    virtual ~TaskQueue() = default;

    virtual void barrier() = 0;

    virtual void copyToHost(MemoryView &mv) = 0;
    virtual void copyToDevice(MemoryView &mv) = 0;

    virtual Future *launch(Kernel &k, MemoryView *params, size_t dim0, size_t dim1, size_t dim2) = 0;

    virtual void sync() = 0;
};

} // namespace ispcrt
