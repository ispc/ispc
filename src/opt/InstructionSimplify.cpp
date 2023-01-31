/*
  Copyright (c) 2022-2023, Intel Corporation
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

#include "InstructionSimplify.h"

namespace ispc {

char InstructionSimplifyPass::ID = 0;

static llvm::Value *lSimplifyBoolVec(llvm::Value *value) {
    llvm::TruncInst *trunc = llvm::dyn_cast<llvm::TruncInst>(value);
    if (trunc != NULL) {
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
    if (icmp != NULL) {
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
    return NULL;
}

static bool lSimplifySelect(llvm::SelectInst *selectInst, llvm::BasicBlock::iterator iter) {
    if (selectInst->getType()->isVectorTy() == false)
        return false;
    Assert(selectInst->getOperand(1) != NULL);
    Assert(selectInst->getOperand(2) != NULL);
    llvm::Value *factor = selectInst->getOperand(0);

    // Simplify all-on or all-off mask values
    MaskStatus maskStatus = GetMaskStatusFromValue(factor);
    llvm::Value *value = NULL;
    if (maskStatus == MaskStatus::all_on)
        // Mask all on -> replace with the first select value
        value = selectInst->getOperand(1);
    else if (maskStatus == MaskStatus::all_off)
        // Mask all off -> replace with the second select value
        value = selectInst->getOperand(2);
    if (value != NULL) {
        ReplaceInstWithValueWrapper(iter, value);
        return true;
    }

    // Sometimes earlier LLVM optimization passes generate unnecessarily
    // complex expressions for the selection vector, which in turn confuses
    // the code generators and leads to sub-optimal code (particularly for
    // 8 and 16-bit masks).  We'll try to simplify them out here so that
    // the code generator patterns match..
    if ((factor = lSimplifyBoolVec(factor)) != NULL) {
        llvm::Instruction *newSelect = llvm::SelectInst::Create(factor, selectInst->getOperand(1),
                                                                selectInst->getOperand(2), selectInst->getName());
        llvm::ReplaceInstWithInst(selectInst, newSelect);
        return true;
    }

    return false;
}

static bool lSimplifyCall(llvm::CallInst *callInst, llvm::BasicBlock::iterator iter) {
    llvm::Function *calledFunc = callInst->getCalledFunction();

    // Turn a __movmsk call with a compile-time constant vector into the
    // equivalent scalar value.
    if (calledFunc == NULL || calledFunc != m->module->getFunction("__movmsk"))
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

restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::SelectInst *selectInst = llvm::dyn_cast<llvm::SelectInst>(&*iter);
        if (selectInst && lSimplifySelect(selectInst, iter)) {
            modifiedAny = true;
            goto restart;
        }
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst && lSimplifyCall(callInst, iter)) {
            modifiedAny = true;
            goto restart;
        }
    }

    DEBUG_END_BB("InstructionSimplify");

    return modifiedAny;
}

bool InstructionSimplifyPass::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("InstructionSimplifyPass::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= simplifyInstructions(BB);
    }
    return modifiedAny;
}

llvm::Pass *CreateInstructionSimplifyPass() { return new InstructionSimplifyPass; }

} // namespace ispc
