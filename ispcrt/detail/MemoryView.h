// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "IntrusivePtr.h"

namespace ispcrt {
namespace base {

struct Device;

struct MemoryView : public RefCounted {
    MemoryView() = default;
    virtual ~MemoryView() = default;

    virtual bool isShared() = 0;
    virtual void *hostPtr() = 0;
    virtual void *devicePtr() = 0;
    virtual size_t numBytes() = 0;
};

} // namespace base
} // namespace ispcrt
