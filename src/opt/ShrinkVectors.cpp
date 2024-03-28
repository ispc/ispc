/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ShrinkVectors.h"

#include <llvm/IR/IRBuilder.h>

namespace ispc {

bool lIsSecondHalfAllFalse(llvm::Value *mask) {
    if (auto *CV = llvm::dyn_cast<llvm::ConstantVector>(mask)) {
        auto N = CV->getType()->getNumElements();
        for (auto i = N / 2; i < N; i++) {
            llvm::Constant *E = CV->getAggregateElement(i);
            if (!E || !llvm::isa<llvm::ConstantInt>(E) || !llvm::cast<llvm::ConstantInt>(E)->isZero()) {
                // found non-zero element
                return false;
            }
        }
        return true;
    }
    return false;
}

llvm::Value *lExtendVector(llvm::IRBuilder<> &B, llvm::Value *originalVector) {
    auto *vecType = llvm::cast<llvm::VectorType>(originalVector->getType());
    unsigned N = vecType->getElementCount().getKnownMinValue();

    // Create a vector of indices to select the first half of the elements
    std::vector<llvm::Constant *> indices;
    for (unsigned i = 0; i < N * 2; ++i) {
        indices.push_back(llvm::ConstantInt::get(B.getInt32Ty(), i));
    }
    llvm::ArrayRef<llvm::Constant *> indicesRef(indices);
    llvm::Constant *mask = llvm::ConstantVector::get(indicesRef);

    // Use shufflevector instruction to create the new vector
    return B.CreateShuffleVector(originalVector, llvm::UndefValue::get(vecType), mask);
}

llvm::Value *lShrinkVector(llvm::IRBuilder<> &B, llvm::Value *originalVector) {
    auto *vecType = llvm::cast<llvm::VectorType>(originalVector->getType());
    unsigned N = vecType->getElementCount().getKnownMinValue();

    // Create a vector of indices to select the first half of the elements
    std::vector<llvm::Constant *> indices;
    for (unsigned i = 0; i < N / 2; ++i) {
        indices.push_back(llvm::ConstantInt::get(B.getInt32Ty(), i));
    }
    llvm::ArrayRef<llvm::Constant *> indicesRef(indices);
    llvm::Constant *mask = llvm::ConstantVector::get(indicesRef);

    // Use shufflevector instruction to create the new vector
    return B.CreateShuffleVector(originalVector, llvm::UndefValue::get(vecType), mask);
}

llvm::Value *lBitcastPointerType(llvm::IRBuilder<> &B, llvm::Value *ptr, llvm::Value *value) {
    auto *vecType = llvm::cast<llvm::VectorType>(value->getType());
    auto *newPtrType = llvm::PointerType::get(vecType, 0 /* TODO! */);
    // TODO! opaque pointer is no-op here, any special handling?
    return B.CreateBitCast(ptr, newPtrType);
}

llvm::Constant *lShrinkConstVec(llvm::LLVMContext &context, llvm::Value *originalValue) {
    llvm::Constant *originalMask = llvm::cast<llvm::Constant>(originalValue);
    auto *vecType = llvm::cast<llvm::VectorType>(originalMask->getType());
    unsigned N = vecType->getElementCount().getKnownMinValue();
    unsigned newSize = N / 2;

    // Extract the first half of the elements
    std::vector<llvm::Constant *> newValues;
    for (unsigned i = 0; i < newSize; ++i) {
        auto *element = originalMask->getAggregateElement(i);
        if (!element) {
            // Handle error in extracting element
            return nullptr;
        }
        newValues.push_back(element);
    }

    // Create a new constant vector with the first half of the elements
    llvm::ArrayRef<llvm::Constant *> newValuesRef(newValues);
    return llvm::ConstantVector::get(newValuesRef);
}

// This function replaces, e.g.,
//
// call void @llvm.masked.store.v8f32.p0(<8 x float> %value, ptr %10, i32 1,
//                          <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false>)
//
// with
//
// %half_value = shufflevector <8 x float> %value, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// call void @llvm.masked.store.v4f32.p0(<4 x float> %half_value, ptr %10, i32 1,
//                                       <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
//
void lReplaceMaskedStore(llvm::IRBuilder<> &B, std::deque<llvm::Value *> &WL, llvm::CallInst *CI) {
    llvm::Function *F = CI->getCalledFunction();
    llvm::Value *value = CI->getOperand(0);
    llvm::Value *ptr = CI->getOperand(1);
    int alignment = llvm::dyn_cast<llvm::ConstantInt>(CI->getOperand(2))->getZExtValue();
    llvm::Value *mask = CI->getOperand(3);

    WL.push_back(value);

    B.SetInsertPoint(CI);
    value = lShrinkVector(B, value);
    ptr = lBitcastPointerType(B, ptr, value);
    mask = lShrinkConstVec(F->getParent()->getContext(), mask);
    auto *newCall = B.CreateMaskedStore(value, ptr, llvm::Align(alignment), mask);

    LLVMCopyMetadata(newCall, CI);
    CI->eraseFromParent();
}

// This function replaces, e.g.,
//
// %6 = call <8 x float> @llvm.masked.load.v8f32.p0(ptr %5, i32 1,
//      <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false>,
//      <8 x float> <float poison, float poison, float poison, float poison, float 0, float 0, float 0, float 0>)
//
// with
//
// %7 = call <4 x float> @llvm.masked.load.v4f32.p0(ptr %6, i32 1,
//                                    <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x float> poison)
// %8 = shufflevector <4 x float> %7, <4 x float> undef,
//                                    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
//
void lReplaceMaskedLoad(llvm::IRBuilder<> &B, std::deque<llvm::Value *> &WL, llvm::CallInst *CI) {
    llvm::Function *F = CI->getCalledFunction();
    llvm::Value *ptr = CI->getOperand(0);
    llvm::Value *alignment = CI->getOperand(1);
    llvm::Value *mask = CI->getOperand(2);
    llvm::Value *passthrough = CI->getOperand(3);

    B.SetInsertPoint(CI);

    passthrough = lShrinkConstVec(F->getParent()->getContext(), passthrough);
    ptr = lBitcastPointerType(B, ptr, passthrough);

    assert(lIsSecondHalfAllFalse(mask));
    mask = lShrinkConstVec(F->getParent()->getContext(), mask);

    auto *vecType = passthrough->getType();
    auto *ptrType = llvm::PointerType::get(vecType, 0 /* TODO! */);
    llvm::Function *func =
        llvm::Intrinsic::getDeclaration(F->getParent(), llvm::Intrinsic::masked_load, {vecType, ptrType});

    llvm::Value *args[] = {ptr, alignment, mask, passthrough};
    llvm::Value *newLoad = B.CreateCall(func, args);
    llvm::Value *replacement = lExtendVector(B, newLoad);

    LLVMCopyMetadata(newLoad, CI);
    CI->replaceAllUsesWith(replacement);
    CI->eraseFromParent();
}

// This function replaces, e.g.,
//
// %9 = tail call <8 x float> @llvm.x86.avx.max.ps.256(<8 x float> %6, <8 x float> %8)
//
// with
//
// %13 = shufflevector <8 x float> %8, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// %14 = shufflevector <8 x float> %12, <8 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// %15 = call <4 x float> @llvm.x86.sse.max.ps(<4 x float> %13, <4 x float> %14)
// %16 = shufflevector <4 x float> %15, <4 x float> undef,
//                                      <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
//
void lReplaceMax(llvm::IRBuilder<> &B, std::deque<llvm::Value *> &WL, llvm::CallInst *CI) {
    llvm::Function *F = CI->getCalledFunction();
    llvm::Value *op0 = CI->getOperand(0);
    llvm::Value *op1 = CI->getOperand(1);

    WL.push_back(op0);
    WL.push_back(op1);

    B.SetInsertPoint(CI);
    llvm::Value *newOp0 = lShrinkVector(B, op0);
    llvm::Value *newOp1 = lShrinkVector(B, op1);
    auto *newType = newOp0->getType();

    auto intrinsicID = llvm::Intrinsic::x86_sse_max_ps;
    llvm::FunctionCallee intrinsicFunc =
        F->getParent()->getOrInsertFunction(llvm::Intrinsic::getName(intrinsicID), newType, newType, newType);
    llvm::Value *args[] = {newOp0, newOp1};
    llvm::CallInst *newCall = B.CreateCall(intrinsicFunc, args);

    // Just to be simple, extend the half vector back to original size to
    // extend the old uses. We rely on later optimization to remove theses
    // extra extending and shrinking instruction.
    llvm::Value *replacement = lExtendVector(B, newCall);

    LLVMCopyMetadata(newCall, CI);
    CI->replaceAllUsesWith(replacement);
    CI->eraseFromParent();
}

llvm::PreservedAnalyses ShrinkVectorsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::Function *f = &F;
    std::vector<llvm::CallInst *> storesToReplace;
    if (f != nullptr && f->empty() == false) {
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
                    llvm::Function *F = CI->getCalledFunction();
                    if (F && F->isIntrinsic() && F->getIntrinsicID() == llvm::Intrinsic::masked_store) {
                        llvm::Value *mask = CI->getOperand(3);
                        if (lIsSecondHalfAllFalse(mask)) {
                            // TODO! sane debug output
                            // TODO! remarks
                            // printf("SHRINK masked.store %s\n", f->getName().str().c_str());
                            storesToReplace.push_back(CI);
                        }
                    }
                }
            }
        }
    }

    // First, we replace masked store with more narrow ones. This store
    // replacement on its own is a good thing to do, then the fact that we
    // don't need the second half of vector register can be propagated further
    // (to uses) to shrink other operations to half width vectors. Then stored
    // value is added to the work list to be processed if possible below in
    // regard to masked loads and some other instructions. We know that this
    // value has the unused second part. We will propagate this through
    // all instructions to the loads (ideally) .
    llvm::IRBuilder<> builder(F.getParent()->getContext());
    std::deque<llvm::Value *> worklist;
    for (auto call : storesToReplace) {
        lReplaceMaskedStore(builder, worklist, call);
    }

    if (!worklist.empty()) {
        // printf("Propagate further\n");
    }
    while (!worklist.empty()) {
        llvm::Value *val = worklist.front();
        worklist.pop_front();
        if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(val)) {
            llvm::Function *func = call->getCalledFunction();
            assert(func);

            // TODO! We may probably want to consider more
            // instruction/intrinsics to propagate shrinking fact.
            if (func->isIntrinsic()) {
                switch (func->getIntrinsicID()) {
                case llvm::Intrinsic::x86_avx_max_ps_256:
                    lReplaceMax(builder, worklist, call);
                    break;
                case llvm::Intrinsic::masked_load:
                    lReplaceMaskedLoad(builder, worklist, call);
                    break;
                }
            }
        }
        // If we don't find any rules to further propagation then it is fine
        // until we do local transformation above.
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
