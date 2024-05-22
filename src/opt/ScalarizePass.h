/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {

// ScalarizePass change some vector operations to scalar ones.
// There are patterns with arithmetic operations with vectors that fully
// consist of undef/poison values except the first element. In this case,
// it can be replaced with a scalar operation with the following broadcast.
// This can help to reduce the number of vector operations and reduce the
// amount of used vector registers. It is especially useful for XE targets
// where such pattern happen often. Although, it can be useful for CPU targets
// as well. This currently covering binary operations with such values.
// More details can be found in the test/lit-tests/scalarize.ll file.

struct ScalarizePass : public llvm::PassInfoMixin<ScalarizePass> {

    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool matchAndReplace(llvm::BasicBlock &BB);
};

} // namespace ispc
