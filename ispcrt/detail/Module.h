// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "IntrusivePtr.h"

namespace ispcrt {
namespace base {

struct Module : public RefCounted {
    Module() = default;
    virtual ~Module() = default;
};

} // namespace base
} // namespace ispcrt
