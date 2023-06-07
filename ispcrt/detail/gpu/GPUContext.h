// Copyright 2022-2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include "../Context.h"
#include <memory>

namespace ispcrt {
namespace gpu {
class ChunkedPool;
}

struct GPUContext : public ispcrt::base::Context {
    GPUContext();
    GPUContext(void *nativeContext);
    ~GPUContext();
    base::MemoryView *newMemoryView(void *appMem, size_t numBytes,
                                    const ISPCRTNewMemoryViewFlags *flags) const override;
    ISPCRTDeviceType getDeviceType() const override;

    virtual void *contextNativeHandle() const override;
    gpu::ChunkedPool *memPool(ISPCRTSharedMemoryAllocationHint type) const;

  private:
    void *m_context{nullptr};
    void *m_driver{nullptr};
    bool m_is_mock{false};
    bool m_has_context_ownership{true};
    std::unique_ptr<gpu::ChunkedPool> m_memPoolHWDR;
    std::unique_ptr<gpu::ChunkedPool> m_memPoolHRDW;
};
} // namespace ispcrt

// Expose API of GPU device solib for dlsym.
extern "C" {
ispcrt::base::Context *load_gpu_context();
ispcrt::base::Context *load_gpu_context_ctx(void *ctx);
}
