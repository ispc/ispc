/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {

/** This pass checks IR for Xe target and fix arguments for Xe intrinsics if needed.
    In case if unsupported statement is found, it reports error and stops compilation.
    Currently it performs 2 checks:
    1. double type support by target
    2. prefetch support by target and fixing prefetch args
 */

class CheckIRForXeTarget : public llvm::FunctionPass {
  public:
    static char ID;
    explicit CheckIRForXeTarget() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "Check and fix IR for Xe target"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool checkAndFixIRForXe(llvm::BasicBlock &BB);
};

llvm::Pass *CreateCheckIRForXeTarget();

} // namespace ispc

#endif
