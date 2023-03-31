/*
  Copyright (c) 2022, Intel Corporation
  All rights reserved.

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#ifdef ISPC_XE_ENABLED
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>

namespace ispc {

/** This pass mangles SPIR-V OpenCL builtins used in Xe target file
 */

class MangleOpenCLBuiltins : public llvm::FunctionPass {
  public:
    static char ID;
    explicit MangleOpenCLBuiltins() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "Mangle OpenCL builtins"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool mangleOpenCLBuiltins(llvm::BasicBlock &BB);
};

llvm::Pass *CreateMangleOpenCLBuiltins();

} // namespace ispc

#endif