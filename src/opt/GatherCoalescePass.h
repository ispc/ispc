/*
  Copyright (c) 2022-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

#include <memory>

namespace ispc {

class MemoryPlanDump;
struct MemoryPlanIds;

// This pass implements two optimizations to improve the performance of
// gathers; currently only gathers of 32-bit values where it can be
// determined at compile time that the mask is all on are supported, though
// both of those limitations may be generalized in the future.
//
//  First, for any single gather, see if it's worthwhile to break it into
//  any of scalar, 2-wide (i.e. 64-bit), 4-wide, or 8-wide loads.  Further,
//  we generate code that shuffles these loads around.  Doing fewer, larger
//  loads in this manner, when possible, can be more efficient.
//
//  Second, this pass can coalesce memory accesses across multiple
//  gathers. If we have a series of gathers without any memory writes in
//  the middle, then we try to analyze their reads collectively and choose
//  an efficient set of loads for them.  Not only does this help if
//  different gathers reuse values from the same location in memory, but
//  it's specifically helpful when data with AOS layout is being accessed;
//  in this case, we're often able to generate wide vector loads and
//  appropriate shuffles automatically.

struct GatherCoalescePass : public llvm::PassInfoMixin<GatherCoalescePass> {
    /** The pass's number in the optimization pipeline, for the lines of
        --dump-memory-plans; set by DebugModulePassManager, 0 elsewhere. */
    int memoryPlanStage = 0;
    /** The ids this pass has seen in the module, for --dump-memory-plans. */
    std::shared_ptr<MemoryPlanIds> memoryPlanIds;

    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    // Type of base pointer element type (the 1st argument of the intrinsic) is i8
    // e.g. @__pseudo_gather_factored_base_offsets32_i32(i8 *, <WIDTH x i32>, i32, <WIDTH x i32>, <WIDTH x MASK>)
    llvm::Type *baseType{LLVMTypes::Int8Type};
    bool coalesceGathersFactored(llvm::BasicBlock &BB, MemoryPlanDump *dump);
};

} // namespace ispc
