// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "Fence.h"
#include "Future.h"
#include "IntrusivePtr.h"
#include "Kernel.h"
#include "MemoryView.h"

namespace ispcrt {
namespace base {

struct CommandList : public RefCounted {
    CommandList() = default;
    virtual ~CommandList() = default;

    virtual void barrier() = 0;
    virtual base::Future *copyToHost(base::MemoryView &mv) = 0;
    virtual base::Future *copyToDevice(base::MemoryView &mv) = 0;
    virtual base::Future *copyMemoryView(base::MemoryView &mv_dst, base::MemoryView &mv_src, const size_t size) = 0;
    virtual base::Future *launch(Kernel &k, base::MemoryView *params, size_t dim0, size_t dim1, size_t dim2) = 0;

    virtual void close() = 0;
    virtual base::Fence *submit() = 0;
    virtual void reset() = 0;

    virtual void enableTimestamps() = 0;
    virtual void *nativeHandle() const = 0;
};

} // namespace base
} // namespace ispcrt
