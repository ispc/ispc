/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

/** For any gathers and scatters remaining after the GSToLoadStorePass
    runs, we need to turn them into actual native gathers and scatters.
    This task is handled by the ReplacePseudoMemoryOpsPass here.
 */
class ReplacePseudoMemoryOpsPass : public llvm::FunctionPass {
  public:
    static char ID;
    explicit ReplacePseudoMemoryOpsPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "Replace Pseudo Memory Ops"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool replacePseudoMemoryOps(llvm::BasicBlock &BB);
};

llvm::Pass *CreateReplacePseudoMemoryOpsPass();

} // namespace ispc
