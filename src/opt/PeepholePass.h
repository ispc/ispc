/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

// PeepholePass

struct PeepholePass : public llvm::PassInfoMixin<PeepholePass> {

    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool matchAndReplace(llvm::BasicBlock &BB);
};

} // namespace ispc
