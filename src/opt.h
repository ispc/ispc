/*
  Copyright (c) 2010-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file opt.h
    @brief Declarations related to optimization passes
*/

#pragma once

#include "ispc.h"

namespace ispc {

/** Optimize the functions in the given module, applying the specified
    level of optimization.  optLevel zero corresponds to essentially no
    optimization--just enough to generate correct code, while level one
    corresponds to full optimization.
*/
void Optimize(llvm::Module *module, int optLevel);
} // namespace ispc
