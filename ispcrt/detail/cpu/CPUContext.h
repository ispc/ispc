// Copyright 2022-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Context.h"

namespace ispcrt {

struct CPUContext : public base::Context {
    CPUContext() = default;
    base::MemoryView *newMemoryView(void *appMem, size_t numBytes,
                                    const ISPCRTNewMemoryViewFlags *flags) const override;
    ISPCRTDeviceType getDeviceType() const override;

    virtual void *contextNativeHandle() const override;
};

} // namespace ispcrt

// Expose API of CPU device solib for dlsym.
extern "C" {
ispcrt::base::Context *load_cpu_context();
}
