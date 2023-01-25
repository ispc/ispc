/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

/** This simple optimization pass looks for a vector select instruction
    with an all-on or all-off constant mask, simplifying it to the
    appropriate operand if so.

    @todo The better thing to do would be to submit a patch to LLVM to get
    these; they're presumably pretty simple patterns to match.
*/
class InstructionSimplifyPass : public llvm::PassInfoMixin<InstructionSimplifyPass> {
  public:
    explicit InstructionSimplifyPass() {}

    static llvm::StringRef getPassName() { return "Vector Select Optimization"; }
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool simplifyInstructions(llvm::BasicBlock &BB);
};

} // namespace ispc
