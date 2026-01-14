/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

struct LowerAMXBuiltinsPass : public llvm::PassInfoMixin<LowerAMXBuiltinsPass> {
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace ispc
