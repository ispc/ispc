// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"
// internal
#include "IntrusivePtr.h"

namespace ispcrt {
namespace base {

struct ModuleOptions : public RefCounted {
    ModuleOptions() = default;
    virtual ~ModuleOptions() = default;
    virtual uint32_t stackSize() const = 0;
    virtual bool libraryCompilation() const = 0;
    virtual ISPCRTModuleType moduleType() const = 0;
    virtual void setStackSize(uint32_t) = 0;
    virtual void setLibraryCompilation(bool) = 0;
    virtual void setModuleType(ISPCRTModuleType) = 0;
};

} // namespace base
} // namespace ispcrt
