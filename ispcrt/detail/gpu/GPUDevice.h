// Copyright Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Device.h"
#include "../Future.h"

namespace ispcrt {

struct GPUDevice : public base::Device {
    GPUDevice();

    base::MemoryView *newMemoryView(void *appMem, size_t numBytes) const override;

    base::TaskQueue *newTaskQueue() const override;

    base::Module *newModule(const char *moduleFile) const override;

    base::Kernel *newKernel(const base::Module &module, const char *name) const override;

  private:
    void *m_driver{nullptr};
    void *m_device{nullptr};
    void *m_context{nullptr};
    bool  m_is_mock{false};
};

} // namespace ispcrt
