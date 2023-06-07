// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "IntrusivePtr.h"

namespace ispcrt {
namespace base {

struct Fence : public RefCounted {
    Fence() = default;
    virtual ~Fence() = default;

    virtual void sync() = 0;
    virtual ISPCRTFenceStatus status() const = 0;
    virtual void reset() = 0;

    virtual void *nativeHandle() const = 0;
};

} // namespace base
} // namespace ispcrt
