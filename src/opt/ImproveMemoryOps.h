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

/** When the front-end emits gathers and scatters, it generates an array of
    vector-width pointers to represent the set of addresses to read from or
    write to.  This optimization detects cases when the base pointer is a
    uniform pointer or when the indexing is into an array that can be
    converted into scatters/gathers from a single base pointer and an array
    of offsets.

    See for example the comments discussing the __pseudo_gather functions
    in builtins.cpp for more information about this.
 */
struct ImproveMemoryOpsPass : public llvm::PassInfoMixin<ImproveMemoryOpsPass> {
    /** The pass's number in the optimization pipeline, for the lines of
        --dump-memory-plans; set by DebugModulePassManager, 0 elsewhere. */
    int memoryPlanStage = 0;
    /** The ids this pass has seen in the module, for --dump-memory-plans. */
    std::shared_ptr<MemoryPlanIds> memoryPlanIds;

    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool improveMemoryOps(llvm::BasicBlock &BB, MemoryPlanDump *dump);
};

} // namespace ispc
