/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

/** When the front-end emits gathers and scatters, it generates an array of
    vector-width pointers to represent the set of addresses to read from or
    write to.  This optimization detects cases when the base pointer is a
    uniform pointer or when the indexing is into an array that can be
    converted into scatters/gathers from a single base pointer and an array
    of offsets.

    See for example the comments discussing the __pseudo_gather functions
    in builtins.cpp for more information about this.
 */
class ImproveMemoryOpsPass : public llvm::FunctionPass {
  public:
    static char ID;
    explicit ImproveMemoryOpsPass() : FunctionPass(ID) {}

    llvm::StringRef getPassName() const override { return "Improve Memory Ops"; }
    bool runOnFunction(llvm::Function &F) override;

  private:
    bool improveMemoryOps(llvm::BasicBlock &BB);
};

llvm::Pass *CreateImproveMemoryOpsPass();

} // namespace ispc
