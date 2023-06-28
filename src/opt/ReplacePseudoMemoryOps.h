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
class ReplacePseudoMemoryOpsPass : public llvm::PassInfoMixin<ReplacePseudoMemoryOpsPass> {
  public:
    explicit ReplacePseudoMemoryOpsPass() {}

    static llvm::StringRef getPassName() { return "Replace Pseudo Memory Ops"; }
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool replacePseudoMemoryOps(llvm::BasicBlock &BB);
};

} // namespace ispc
