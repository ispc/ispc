/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {
// This pass refines vector operations by transitioning from wider vectors to
// narrow ones, e.g., from 8-element to 4-element vectors. It identifies masked
// stores related to 8-element vectors with the second half being redundant.
// Such stores are queued for replacement with their 4-element counterparts,
// which is then followed by adjustments in the surrounding code to maintain
// consistency.

class ShrinkVectorsPass : public llvm::PassInfoMixin<ShrinkVectorsPass> {
  public:
    explicit ShrinkVectorsPass() {}

    static llvm::StringRef getPassName() { return "Shrink vectors"; }
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace ispc
