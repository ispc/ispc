/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ReplaceStdlibShiftPass.h"

namespace ispc {

/** Given an llvm::Value known to be an integer, return its value as
    an int64_t.
*/
static int64_t lGetIntValue(llvm::Value *offset) {
    llvm::ConstantInt *intOffset = llvm::dyn_cast<llvm::ConstantInt>(offset);
    Assert(intOffset && (intOffset->getBitWidth() == 32 || intOffset->getBitWidth() == 64));
    return intOffset->getSExtValue();
}

// This pass replaces shift() with ShuffleVector when the offset is a constant.
// rotate() which is similar in functionality has a slightly different
// implementation. This is due to LLVM(createInstructionCombiningPass)
// optimizing rotate() implementation better when similar implementations
// are used for both. This is a hack to produce similarly optimized code for
// shift.
bool ReplaceStdlibShiftPass::replaceStdlibShiftBuiltin(llvm::BasicBlock &bb) {
    DEBUG_START_BB("ReplaceStdlibShiftPass");
    bool modifiedAny = false;

    llvm::Module *M = bb.getModule();
    llvm::Function *shifts[6];
    shifts[0] = M->getFunction("shift___vytuni");
    shifts[1] = M->getFunction("shift___vysuni");
    shifts[2] = M->getFunction("shift___vyiuni");
    shifts[3] = M->getFunction("shift___vyIuni");
    shifts[4] = M->getFunction("shift___vyfuni");
    shifts[5] = M->getFunction("shift___vyduni");

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::Instruction *inst = &*(iter++);

        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            for (int i = 0; i < 6; i++) {
                if (shifts[i] && (shifts[i] == func)) {
                    // we matched a call
                    llvm::Value *shiftedVec = ci->getArgOperand(0);
                    llvm::Value *shiftAmt = ci->getArgOperand(1);
                    if (llvm::isa<llvm::Constant>(shiftAmt)) {
                        int vectorWidth = g->target->getVectorWidth();
                        int *shuffleVals = new int[vectorWidth];
                        int shiftInt = lGetIntValue(shiftAmt);
                        for (int i = 0; i < vectorWidth; i++) {
                            int s = i + shiftInt;
                            s = (s < 0) ? vectorWidth : s;
                            s = (s >= vectorWidth) ? vectorWidth : s;
                            shuffleVals[i] = s;
                        }
                        llvm::Value *shuffleIdxs = LLVMInt32Vector(shuffleVals);
                        llvm::Value *zeroVec = llvm::ConstantAggregateZero::get(shiftedVec->getType());
                        llvm::Value *shuffle =
                            new llvm::ShuffleVectorInst(shiftedVec, zeroVec, shuffleIdxs, "vecShift", ci);
                        ci->replaceAllUsesWith(shuffle);
                        modifiedAny = true;
                        delete[] shuffleVals;
                    } else if (g->opt.level > 0) {
                        PerformanceWarning(SourcePos(), "Stdlib shift() called without constant shift amount.");
                    }
                }
            }
        }
    }

    DEBUG_END_BB("ReplaceStdlibShiftPass");

    return modifiedAny;
}

llvm::PreservedAnalyses ReplaceStdlibShiftPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("ReplaceStdlibShiftPass::run", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= replaceStdlibShiftBuiltin(BB);
    }
    if (!modifiedAny) {
        // No changes, all analyses are preserved.
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
