/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {
/** There are a number of target-specific functions that we use during these
    optimization passes. We may need to generate calls to them during later
    optimization passes, e.g., ImproveMemoryOps and ReplacePseudoMemoryOps
    passes. To preserve them, we just reference these functions in
    llvm.compiler.used symbol to preserve them across GlobalDCE runs. By the
    time we are done with optimization, any uses of these should be inlined and
    no calls to these functions should remain. This pass deletes
    llvm.compiler.used symbol from the module effectively removing last uses
    of these functions, so that subsequent passes can eliminate them as dead
    code, thus cleaning up the final code output by the compiler.
 */
class RemovePersistentFuncsPass : public llvm::PassInfoMixin<RemovePersistentFuncsPass> {
  public:
    explicit RemovePersistentFuncsPass() {}

    static llvm::StringRef getPassName() { return "Remove persistent function"; }
    llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace ispc
