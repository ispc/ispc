/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "RestoreInlineAttrPass.h"

namespace ispc {

llvm::PreservedAnalyses RestoreInlineAttrPass::run(llvm::Module &M, [[maybe_unused]] llvm::ModuleAnalysisManager &MAM) {
    bool changed = false;
    for (llvm::Function &F : M) {
        if (F.hasFnAttribute("ispc-defer-alwaysinline")) {
            F.removeFnAttr("ispc-defer-alwaysinline");
            // A user-written `noinline` should win over `inline`. The two
            // qualifiers cannot appear together in source, but a function
            // could acquire NoInline from another path (e.g. a redeclaration);
            // be conservative and skip in that case.
            if (!F.hasFnAttribute(llvm::Attribute::NoInline)) {
                F.addFnAttr(llvm::Attribute::AlwaysInline);
            }
            changed = true;
        }
    }

    if (!changed) {
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
