// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "IntrusivePtr.h"

namespace ispcrt {

struct Future : public RefCounted {
    uint64_t time{0};
    bool valid{false};

    Future() = default;
    virtual ~Future() = default;
};

} // namespace ispcrt
