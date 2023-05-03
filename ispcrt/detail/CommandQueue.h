// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "CommandList.h"
#include "IntrusivePtr.h"

namespace ispcrt {
namespace base {

struct CommandQueue : public RefCounted {
    CommandQueue() = default;
    virtual ~CommandQueue() = default;

    virtual base::CommandList *createCommandList() = 0;
    virtual void sync() = 0;

    virtual void *nativeHandle() const = 0;
};

} // namespace base
} // namespace ispcrt
