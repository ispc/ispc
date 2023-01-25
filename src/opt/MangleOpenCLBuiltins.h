/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#ifdef ISPC_XE_ENABLED
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>

namespace ispc {

/** This pass mangles SPIR-V OpenCL builtins used in Xe target file
 */

class MangleOpenCLBuiltins : public llvm::PassInfoMixin<MangleOpenCLBuiltins> {
  public:
    explicit MangleOpenCLBuiltins() {}

    static llvm::StringRef getPassName() { return "Mangle OpenCL builtins"; }
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool mangleOpenCLBuiltins(llvm::BasicBlock &BB);
};

} // namespace ispc

#endif
