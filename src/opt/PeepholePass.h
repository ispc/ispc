/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

// PeepholePass

class PeepholePass : public llvm::PassInfoMixin<PeepholePass> {
  public:
    explicit PeepholePass(){};

    static llvm::StringRef getPassName() { return "Peephole Optimizations"; }
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool matchAndReplace(llvm::BasicBlock &BB);
};

} // namespace ispc
