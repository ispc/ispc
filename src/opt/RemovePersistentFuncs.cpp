/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "RemovePersistentFuncs.h"
#include "builtins-decl.h"

namespace ispc {

using namespace builtin;

llvm::PreservedAnalyses RemovePersistentFuncsPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    llvm::GlobalVariable *llvmUsed = M.getNamedGlobal("llvm.compiler.used");
    if (llvmUsed) {
        llvmUsed->eraseFromParent();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
