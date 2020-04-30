// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Device.h"

namespace ispcrt {

struct GPUDevice : public Device {
    GPUDevice();

    MemoryView *newMemoryView(void *appMem, size_t numBytes) const override;

    TaskQueue *newTaskQueue() const override;

    Module *newModule(const char *moduleFile) const override;

    Kernel *newKernel(const Module &module, const char *name) const override;

  private:
    void *m_driver{nullptr};
    void *m_device{nullptr};
};

} // namespace ispcrt
