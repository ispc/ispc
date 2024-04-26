/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

class ReplaceStdlibShiftPass : public llvm::PassInfoMixin<ReplaceStdlibShiftPass> {
  public:
    explicit ReplaceStdlibShiftPass() {}

    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool replaceStdlibShiftBuiltin(llvm::BasicBlock &BB);
};

} // namespace ispc
