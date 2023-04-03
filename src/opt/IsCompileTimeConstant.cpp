/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "IsCompileTimeConstant.h"

namespace ispc {

char IsCompileTimeConstantPass::ID = 0;

bool IsCompileTimeConstantPass::lowerCompileTimeConstant(llvm::BasicBlock &bb) {
    DEBUG_START_BB("IsCompileTimeConstantPass");

    llvm::Function *funcs[] = {m->module->getFunction("__is_compile_time_constant_mask"),
                               m->module->getFunction("__is_compile_time_constant_uniform_int32"),
                               m->module->getFunction("__is_compile_time_constant_varying_int32")};

    bool modifiedAny = false;

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::BasicBlock::iterator curIter = iter++;
        // Iterate through the instructions looking for calls to the
        // __is_compile_time_constant_*() functions
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*(curIter));
        if (callInst == nullptr)
            continue;

        int j;
        int nFuncs = sizeof(funcs) / sizeof(funcs[0]);
        for (j = 0; j < nFuncs; ++j) {
            if (funcs[j] != nullptr && callInst->getCalledFunction() == funcs[j])
                break;
        }
        if (j == nFuncs)
            // not a __is_compile_time_constant_* function
            continue;

        // This optimization pass can be disabled with both the (poorly
        // named) disableGatherScatterFlattening option and
        // disableMaskAllOnOptimizations.
        if (g->opt.disableGatherScatterFlattening || g->opt.disableMaskAllOnOptimizations) {
            ReplaceInstWithValueWrapper(curIter, LLVMFalse);
            modifiedAny = true;
            continue;
        }

        // Is it a constant?  Bingo, turn the call's value into a constant
        // true value.
        llvm::Value *operand = callInst->getArgOperand(0);
        if (llvm::isa<llvm::Constant>(operand)) {
            ReplaceInstWithValueWrapper(curIter, LLVMTrue);
            modifiedAny = true;
            continue;
        }

        // This pass runs multiple times during optimization.  Up until the
        // very last time, it only replaces the call with a 'true' if the
        // value is known to be constant and otherwise leaves the call
        // alone, in case further optimization passes can help resolve its
        // value.  The last time through, it eventually has to give up, and
        // replaces any remaining ones with 'false' constants.
        if (isLastTry) {
            ReplaceInstWithValueWrapper(curIter, LLVMFalse);
            modifiedAny = true;
            continue;
        }
    }

    DEBUG_END_BB("IsCompileTimeConstantPass");

    return modifiedAny;
}

bool IsCompileTimeConstantPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("IsCompileTimeConstantPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= lowerCompileTimeConstant(BB);
    }
    return modifiedAny;
}

llvm::Pass *CreateIsCompileTimeConstantPass(bool isLastTry) { return new IsCompileTimeConstantPass(isLastTry); }

} // namespace ispc
