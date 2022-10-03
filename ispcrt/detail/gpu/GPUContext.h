// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include "../Context.h"

namespace ispcrt {
struct GPUContext : public ispcrt::base::Context {
    GPUContext();
    GPUContext(void* nativeContext);
    ~GPUContext();
    base::MemoryView *newMemoryView(void *appMem, size_t numBytes, bool shared) const override;
    ISPCRTDeviceType getDeviceType() const override;

    virtual void* contextNativeHandle() const override;
private:
    void *m_context{nullptr};
    void *m_driver{nullptr};
    bool  m_is_mock{false};
    bool  m_has_context_ownership{true};
};
}