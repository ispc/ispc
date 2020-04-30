// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "../Device.h"

namespace ispcrt {

struct CPUDevice : public Device {
    CPUDevice() = default;

    MemoryView *newMemoryView(void *appMem, size_t numBytes) const override;

    TaskQueue *newTaskQueue() const override;

    Module *newModule(const char *moduleFile) const override;

    Kernel *newKernel(const Module &module, const char *name) const override;
};

} // namespace ispcrt
