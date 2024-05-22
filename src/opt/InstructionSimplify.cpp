/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "InstructionSimplify.h"
#include "builtins-decl.h"

namespace ispc {

static llvm::Value *lSimplifyBoolVec(llvm::Value *value) {
    llvm::TruncInst *trunc = llvm::dyn_cast<llvm::TruncInst>(value);
    if (trunc != nullptr) {
        // Convert trunc({sext,zext}(i1 vector)) -> (i1 vector)
        llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(value);
        if (sext && sext->getOperand(0)->getType() == LLVMTypes::Int1VectorType)
            return sext->getOperand(0);

        llvm::ZExtInst *zext = llvm::dyn_cast<llvm::ZExtInst>(value);
        if (zext && zext->getOperand(0)->getType() == LLVMTypes::Int1VectorType)
            return zext->getOperand(0);
    }
    /*
      // This optimization has discernable benefit on the perf
      // suite on latest LLVM versions.
      // On 3.4+ (maybe even older), it can result in illegal
      // operations, so it's being disabled.
    llvm::ICmpInst *icmp = llvm::dyn_cast<llvm::ICmpInst>(value);
    if (icmp != nullptr) {
        // icmp(ne, {sext,zext}(foo), zeroinitializer) -> foo
        if (icmp->getSignedPredicate() == llvm::CmpInst::ICMP_NE) {
            llvm::Value *op1 = icmp->getOperand(1);
            if (llvm::isa<llvm::ConstantAggregateZero>(op1)) {
                llvm::Value *op0 = icmp->getOperand(0);
                llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(op0);
                if (sext)
                    return sext->getOperand(0);
                llvm::ZExtInst *zext = llvm::dyn_cast<llvm::ZExtInst>(op0);
                if (zext)
                    return zext->getOperand(0);
            }
        }

    }
    */
    return nullptr;
}

static bool lSimplifySelect(llvm::SelectInst *selectInst, llvm::BasicBlock::iterator iter) {
    if (selectInst->getType()->isVectorTy() == false)
        return false;
    Assert(selectInst->getOperand(1) != nullptr);
    Assert(selectInst->getOperand(2) != nullptr);
    llvm::Value *factor = selectInst->getOperand(0);

    // Simplify all-on or all-off mask values
    MaskStatus maskStatus = GetMaskStatusFromValue(factor);
    llvm::Value *value = nullptr;
    if (maskStatus == MaskStatus::all_on)
        // Mask all on -> replace with the first select value
        value = selectInst->getOperand(1);
    else if (maskStatus == MaskStatus::all_off)
        // Mask all off -> replace with the second select value
        value = selectInst->getOperand(2);
    if (value != nullptr) {
        ReplaceInstWithValueWrapper(iter, value);
        return true;
    }

    // Sometimes earlier LLVM optimization passes generate unnecessarily
    // complex expressions for the selection vector, which in turn confuses
    // the code generators and leads to sub-optimal code (particularly for
    // 8 and 16-bit masks).  We'll try to simplify them out here so that
    // the code generator patterns match..
    if ((factor = lSimplifyBoolVec(factor)) != nullptr) {
        llvm::Instruction *newSelect = llvm::SelectInst::Create(factor, selectInst->getOperand(1),
                                                                selectInst->getOperand(2), selectInst->getName());
        llvm::ReplaceInstWithInst(selectInst, newSelect);
        return true;
    }

    return false;
}

static bool lSimplifyCall(llvm::CallInst *callInst, llvm::BasicBlock::iterator iter) {
    llvm::Function *calledFunc = callInst->getCalledFunction();
    llvm::Module *M = callInst->getModule();

    // Turn a __movmsk call with a compile-time constant vector into the
    // equivalent scalar value.
    if (calledFunc == nullptr || calledFunc != M->getFunction(builtin::__movmsk))
        return false;

    uint64_t mask;
    if (GetMaskFromValue(callInst->getArgOperand(0), &mask) == true) {
        ReplaceInstWithValueWrapper(iter, LLVMInt64(mask));
        return true;
    }
    return false;
}

bool InstructionSimplifyPass::simplifyInstructions(llvm::BasicBlock &bb) {
    DEBUG_START_BB("InstructionSimplify");

    bool modifiedAny = false;

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::BasicBlock::iterator curIter = iter++;
        llvm::SelectInst *selectInst = llvm::dyn_cast<llvm::SelectInst>(&*curIter);
        if (selectInst && lSimplifySelect(selectInst, curIter)) {
            modifiedAny = true;
            continue;
        }
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*curIter);
        if (callInst && lSimplifyCall(callInst, curIter)) {
            modifiedAny = true;
            continue;
        }
    }

    DEBUG_END_BB("InstructionSimplify");

    return modifiedAny;
}

llvm::PreservedAnalyses InstructionSimplifyPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("InstructionSimplifyPass::run", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= simplifyInstructions(BB);
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
