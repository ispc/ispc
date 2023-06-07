// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "IntrusivePtr.h"

namespace ispcrt {
namespace base {

struct Module : public RefCounted {
    Module() = default;
    virtual ~Module() = default;

    virtual void *functionPtr(const char *name) const = 0;
};

} // namespace base
} // namespace ispcrt
