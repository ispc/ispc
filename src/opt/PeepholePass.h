/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

// PeepholePass

class PeepholePass : public llvm::FunctionPass {
  public:
    static char ID;
    explicit PeepholePass() : FunctionPass(ID){};

    llvm::StringRef getPassName() const override { return "Peephole Optimizations"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool matchAndReplace(llvm::BasicBlock &BB);
};

llvm::Pass *CreatePeepholePass();

} // namespace ispc
