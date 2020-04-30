// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "MemoryView.h"

namespace ispcrt {

struct Kernel : public RefCounted {
    Kernel() = default;
    virtual ~Kernel() = default;
};

} // namespace ispcrt
