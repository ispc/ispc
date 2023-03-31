/*
  Copyright (c) 2022-2023, Intel Corporation
  All rights reserved.

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {
/** This pass replaces LLVM intrinsics unsupported on Xe
 */

class ReplaceLLVMIntrinsics : public llvm::FunctionPass {
  public:
    static char ID;
    explicit ReplaceLLVMIntrinsics() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "LLVM intrinsics replacement"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool replaceUnspportedIntrinsics(llvm::BasicBlock &BB);
};

llvm::Pass *CreateReplaceLLVMIntrinsics();

} // namespace ispc

#endif
