// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "MemoryView.h"

namespace ispcrt {
namespace base {

struct Context : public RefCounted {
    Context() = default;
    virtual ~Context() = default;
    virtual MemoryView *newMemoryView(void *appMem, size_t numBytes, const ISPCRTNewMemoryViewFlags *flags) const = 0;
    virtual ISPCRTDeviceType getDeviceType() const = 0;

    virtual void *contextNativeHandle() const = 0;
};

} // namespace base
} // namespace ispcrt
