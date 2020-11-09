// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// public
#include "../ispcrt.h"

// std
#include <stdexcept>
#include <string>

namespace ispcrt {
namespace base {

struct ispcrt_runtime_error : public std::runtime_error {
    ISPCRTError e;

    ispcrt_runtime_error(ISPCRTError _e, const std::string& msg) : std::runtime_error(msg), e {_e} {}
};


} // namespace base
} // namespace ispcrt
