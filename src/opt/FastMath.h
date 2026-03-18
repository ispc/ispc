/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

/** This optimization pass add flags to FP instructions so to enable
    fast-math-related optimizations in the subsequent passes */
struct FastMathPass : public llvm::PassInfoMixin<FastMathPass> {

    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool optimizeFpInstructions(llvm::BasicBlock &BB);
};

} // namespace ispc
