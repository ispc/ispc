/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once

#include "ISPCPass.h"

namespace ispc {
/** This is a relatively simple optimization pass that does a few small
    optimizations that LLVM's x86 optimizer doesn't currently handle.
    (Specifically, MOVMSK of a constant can be replaced with the
    corresponding constant value, BLENDVPS and AVX masked load/store with
    either an 'all on' or 'all off' masks can be replaced with simpler
    operations.

    @todo The better thing to do would be to submit a patch to LLVM to get
    these; they're presumably pretty simple patterns to match.
*/
struct IntrinsicsOpt : public llvm::PassInfoMixin<IntrinsicsOpt> {

    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);

  private:
    bool optimizeIntrinsics(llvm::BasicBlock &BB);

    struct MaskInstruction {
        MaskInstruction(llvm::Function *f) { function = f; }
        llvm::Function *function;
    };
    std::vector<MaskInstruction> maskInstructions;

    /** Structure that records everything we need to know about a blend
        instruction for this optimization pass.
     */
    struct BlendInstruction {
        BlendInstruction(llvm::Function *f, uint64_t ao, int o0, int o1, int of)
            : function(f), allOnMask(ao), op0(o0), op1(o1), opFactor(of) {}
        /** Function pointer for the blend instruction */
        llvm::Function *function;
        /** Mask value for an "all on" mask for this instruction */
        uint64_t allOnMask;
        /** The operand number in the llvm CallInst corresponds to the
            first operand to blend with. */
        int op0;
        /** The operand number in the CallInst corresponding to the second
            operand to blend with. */
        int op1;
        /** The operand in the call inst where the blending factor is
            found. */
        int opFactor;
    };
    std::vector<BlendInstruction> blendInstructions;

    bool matchesMaskInstruction(llvm::Function *function);
    BlendInstruction *matchingBlendInstruction(llvm::Function *function);
};

} // namespace ispc
