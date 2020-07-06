// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "IntrusivePtr.h"

namespace ispcrt {
namespace base {

struct Future : public RefCounted {
    Future() = default;
    virtual ~Future() = default;

    virtual uint64_t time() = 0;
    virtual bool valid() = 0;
};

} // namespace base
} // namespace ispcrt
