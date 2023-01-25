/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {
/** This pass replaces LLVM intrinsics unsupported on Xe
 */

class ReplaceLLVMIntrinsics : public llvm::PassInfoMixin<ReplaceLLVMIntrinsics> {
  public:
    explicit ReplaceLLVMIntrinsics() {}

    static llvm::StringRef getPassName() { return "LLVM intrinsics replacement"; }
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool replaceUnspportedIntrinsics(llvm::BasicBlock &BB);
};

} // namespace ispc

#endif
