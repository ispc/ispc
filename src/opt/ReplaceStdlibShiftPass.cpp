/*
  Copyright (c) 2022, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "ReplaceStdlibShiftPass.h"

namespace ispc {

char ReplaceStdlibShiftPass::ID = 0;

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

    llvm::Function *shifts[6];
    shifts[0] = m->module->getFunction("shift___vytuni");
    shifts[1] = m->module->getFunction("shift___vysuni");
    shifts[2] = m->module->getFunction("shift___vyiuni");
    shifts[3] = m->module->getFunction("shift___vyIuni");
    shifts[4] = m->module->getFunction("shift___vyfuni");
    shifts[5] = m->module->getFunction("shift___vyduni");

    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::Instruction *inst = &*iter;

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

bool ReplaceStdlibShiftPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ReplaceStdlibShiftPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= replaceStdlibShiftBuiltin(BB);
    }
    return modifiedAny;
}

llvm::Pass *CreateReplaceStdlibShiftPass() { return new ReplaceStdlibShiftPass(); }

} // namespace ispc