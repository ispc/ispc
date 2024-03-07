/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

// This pass transforms some masked loads and stores into their unmasked
// counterparts. This is done to improve the performance of the generated code
// in cases when only the part of the vector is actually used. The masked.load
// and masked.store intrinsics are directly mapped to machine instructions with
// the specified full width of vector values being loaded or stored. This
// transformation allows the backend to generate shorter vector memory
// operations and corresponding math operations avoiding extra spills of
// temporal values to memory.

struct ReplaceMaskedMemOpsPass : public llvm::PassInfoMixin<ReplaceMaskedMemOpsPass> {
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace ispc
