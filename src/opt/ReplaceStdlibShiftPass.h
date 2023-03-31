/*
  Copyright (c) 2022-2023, Intel Corporation
  All rights reserved.

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

class ReplaceStdlibShiftPass : public llvm::FunctionPass {
  public:
    static char ID;
    explicit ReplaceStdlibShiftPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "Resolve \"replace extract insert chains\""; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool replaceStdlibShiftBuiltin(llvm::BasicBlock &BB);
};
llvm::Pass *CreateReplaceStdlibShiftPass();
} // namespace ispc
