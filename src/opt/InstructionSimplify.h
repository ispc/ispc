/*
  Copyright (c) 2022, Intel Corporation
  All rights reserved.

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
class InstructionSimplifyPass : public llvm::FunctionPass {
  public:
    static char ID;
    explicit InstructionSimplifyPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "Vector Select Optimization"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool simplifyInstructions(llvm::BasicBlock &BB);
};

llvm::Pass *CreateInstructionSimplifyPass();

} // namespace ispc