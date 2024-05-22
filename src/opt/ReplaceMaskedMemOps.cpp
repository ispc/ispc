/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ReplaceMaskedMemOps.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Local.h>

#include <numeric>

namespace ispc {

bool lIsPowerOf2(unsigned n) { return (n > 0) && !(n & (n - 1)); }

// Check if the mask is a constant vector with the first part being all ones
// and the second part being all zeros. The length of the first part is
// returned in TrueSubmaskLength argument. We are looking for the mask with the
// true prefix of length 2^k only. This is a limitation of the current
// implementation of lMaskedMergeVectors.
bool lCheckMask(llvm::Value *mask, unsigned &TrueSubmaskLength) {
    if (auto *CV = llvm::dyn_cast<llvm::ConstantVector>(mask)) {
        auto N = CV->getType()->getNumElements();
        if (!lIsPowerOf2(N)) {
            return false;
        }

        unsigned TruePrefixLength = 0;
        for (auto i = 0; i < N; i++) {
            llvm::Constant *E = CV->getAggregateElement(i);
            if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(E)) {
                if (C->isOne()) {
                    TruePrefixLength++;
                    continue;
                } else {
                    break;
                }
            }
        }

        for (auto i = TruePrefixLength + 1; i < N; i++) {
            llvm::Constant *E = CV->getAggregateElement(i);
            if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(E)) {
                if (C->isZero()) {
                    continue;
                } else {
                    return false;
                }
            }
        }

        if (!lIsPowerOf2(TruePrefixLength)) {
            return false;
        }

        TrueSubmaskLength = TruePrefixLength;
        return true;
    }
    return false;
}

// Extract the first part of the vector with the shufflevector instruction. The
// length of the part is passed in SubVectorLength.
llvm::Value *lShrinkVector(llvm::IRBuilder<> &B, llvm::Value *originalVector, unsigned SubVectorLength) {
    // Create a vector of indices with [0, 1, 2, ..., SubVectorLength).
    std::vector<unsigned> indices(SubVectorLength);
    std::iota(indices.begin(), indices.end(), 0);
    llvm::Constant *mask = llvm::ConstantDataVector::get(B.getContext(), indices);

    // Use shufflevector instruction to create the new vector
    llvm::Twine name = originalVector->getName() + ".part";
    return B.CreateShuffleVector(originalVector, llvm::UndefValue::get(originalVector->getType()), mask, name);
}

// Create a new vector containing the second part of the const vector.
// The length of the first part is passed in SubVectorLength. The new vector
// has the length of N - SubVectorLength, where N is the length of the original
// vectors we are working with.
llvm::Constant *lExtractSecondPartOfConstVec(llvm::LLVMContext &context, llvm::Constant *originalMask,
                                             unsigned SubVectorLength) {
    auto *vecType = llvm::dyn_cast<llvm::VectorType>(originalMask->getType());
    Assert(vecType);
    unsigned N = vecType->getElementCount().getKnownMinValue();

    // Iterate over the second part of the elements and add them to the new vector.
    std::vector<llvm::Constant *> newValues;
    for (unsigned i = SubVectorLength; i < N; ++i) {
        auto *element = originalMask->getAggregateElement(i);
        if (!element) {
            return nullptr;
        }
        newValues.push_back(element);
    }

    llvm::ArrayRef<llvm::Constant *> newValuesRef(newValues);
    return llvm::ConstantVector::get(newValuesRef);
}

// Merge two vectors into a single vector with the sequence of shufflevector
// instructions. The first vector length is the power of 2 (M). The overall
// length is the power of 2 either (N). This means that we can do the merge in
// the log2(N) - log2(M) steps. The second vector has the length of N. In
// resulting vector, we want to have the first part and the rest elements from
// the second vector. In other words, change in the second vector the first M
// elements with the elemenst from the first vector.
// Ideally, we would like to merge vector of different lengths. This would
// simplify the current implemenation, but shuffle instruction requires the
// vector to have same length.
llvm::Value *lMaskedMergeVectors(llvm::IRBuilder<> &B, llvm::Value *firstVector, llvm::Constant *secondVector,
                                 const llvm::Twine &name) {
    auto *firstVecType = llvm::dyn_cast<llvm::VectorType>(firstVector->getType());
    auto *secondVecType = llvm::dyn_cast<llvm::VectorType>(secondVector->getType());
    Assert(firstVecType && secondVecType);
    unsigned M = firstVecType->getElementCount().getKnownMinValue();
    unsigned N = secondVecType->getElementCount().getKnownMinValue();

    for (unsigned length = M; length < N; length *= 2) {
        // Create a vector of indices [0, 1, 2, ..., length*2).
        std::vector<unsigned> indices(length * 2);
        std::iota(indices.begin(), indices.end(), 0);
        llvm::Constant *mask = llvm::ConstantDataVector::get(B.getContext(), indices);

        // We don't change the second vector here, i.e., it always has the length of N.
        // Slice from the second vector corresponding to [length:N).
        llvm::Value *secondSubVector = lExtractSecondPartOfConstVec(B.getContext(), secondVector, length);
        // Slice from the second vector corresponding to [length:length*2).
        secondSubVector = lShrinkVector(B, secondSubVector, length);

        // Update the first vector, so it has the length of length*2.
        firstVector = B.CreateShuffleVector(firstVector, secondSubVector, mask, name + ".p." + llvm::Twine(length));
    }

    return firstVector;
}

llvm::Value *lBitcastPointerType(llvm::IRBuilder<> &B, llvm::Value *ptr, llvm::Value *value) {
    auto *vecType = llvm::dyn_cast<llvm::VectorType>(value->getType());
    llvm::PointerType *ptrType = llvm::dyn_cast<llvm::PointerType>(ptr->getType());
    Assert(vecType && ptrType);
    auto *newPtrType = llvm::PointerType::get(vecType, ptrType->getAddressSpace());
    // If ptr is opaque pointer then no-op is generated here.
    return B.CreateBitCast(ptr, newPtrType);
}

// This function replaces masked store intrinsic with an unmasked store instruction.
// Unmasked store instruction stores only the first part of the initial vector
// with the length of SubVectorLength.
void lReplaceMaskedStore(llvm::IRBuilder<> &B, llvm::CallInst *CI, unsigned SubVectorLength) {
    llvm::Value *origVec = CI->getOperand(0);
    llvm::Value *ptr = CI->getOperand(1);
    llvm::ConstantInt *alignmentCI = llvm::dyn_cast<llvm::ConstantInt>(CI->getOperand(2));
    Assert(alignmentCI);
    int alignment = alignmentCI->getZExtValue();

    B.SetInsertPoint(CI);

    llvm::Value *subVec = lShrinkVector(B, origVec, SubVectorLength);
    ptr = lBitcastPointerType(B, ptr, subVec);
    llvm::Value *store = B.CreateAlignedStore(subVec, ptr, llvm::Align(alignment));

    LLVMCopyMetadata(store, CI);
    CI->eraseFromParent();
}

// This function replaces half-masked load intrinsic with an unmasked load
// instruction that loads the first part [0:SubVectorLength) of the vector.
// Masked load intrinsic has the passthrough value that is needed to be
// preserved in the new vector in case it is used later. It is done by merging
// the result of the unmasked load with the rest part of the passthrough value.
void lReplaceMaskedLoad(llvm::IRBuilder<> &B, llvm::CallInst *CI, unsigned SubVectorLength) {
    llvm::Value *ptr = CI->getOperand(0);
    llvm::ConstantInt *alignmentCI = llvm::dyn_cast<llvm::ConstantInt>(CI->getOperand(1));
    llvm::Constant *passthrough = llvm::dyn_cast<llvm::Constant>(CI->getOperand(3));
    Assert(alignmentCI && passthrough);
    int alignment = alignmentCI->getZExtValue();

    B.SetInsertPoint(CI);

    auto origName = CI->getName();
    auto *vecType = llvm::dyn_cast<llvm::VectorType>(CI->getType());
    Assert(vecType);
    auto *subVecType = llvm::VectorType::get(vecType->getElementType(), SubVectorLength, false);
    ptr = B.CreateBitCast(ptr, subVecType->getPointerTo());

    llvm::LoadInst *subVec = B.CreateLoad(subVecType, ptr, llvm::Twine(origName) + ".part");
    // It is important to preserve the alignment of the original masked load.
    subVec->setAlignment(llvm::Align(alignment));

    llvm::Value *replacement = lMaskedMergeVectors(B, subVec, passthrough, llvm::Twine(origName));

    LLVMCopyMetadata(replacement, CI);
    CI->replaceAllUsesWith(replacement);
    llvm::RecursivelyDeleteTriviallyDeadInstructions(CI);
}

llvm::PreservedAnalyses ReplaceMaskedMemOpsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    if (F.empty()) {
        return llvm::PreservedAnalyses::all();
    }

    llvm::IRBuilder<> builder(F.getParent()->getContext());
    std::unordered_map<llvm::CallInst *, unsigned> storesToReplace;
    std::unordered_map<llvm::CallInst *, unsigned> loadsToReplace;
    for (auto &BB : F) {
        for (auto &I : BB) {
            auto *CI = llvm::dyn_cast<llvm::CallInst>(&I);
            if (!CI) {
                continue;
            }

            llvm::Function *CF = CI->getCalledFunction();
            if (!(CF && CF->isIntrinsic())) {
                continue;
            }

            unsigned SubVectorLength = 0;
            if (CF->getIntrinsicID() == llvm::Intrinsic::masked_store) {
                llvm::Value *mask = CI->getOperand(3);
                if (lCheckMask(mask, SubVectorLength)) {
                    storesToReplace[CI] = SubVectorLength;
                }
            }

            if (CF->getIntrinsicID() == llvm::Intrinsic::masked_load) {
                llvm::Value *mask = CI->getOperand(2);
                llvm::Value *passthrough = CI->getOperand(3);
                if (llvm::isa<llvm::Constant>(passthrough) && lCheckMask(mask, SubVectorLength)) {
                    loadsToReplace[CI] = SubVectorLength;
                }
            }
        }
    }

    if (storesToReplace.empty() && loadsToReplace.empty()) {
        return llvm::PreservedAnalyses::all();
    }

    for (auto const &[CI, SubVectorLength] : storesToReplace) {
        lReplaceMaskedStore(builder, CI, SubVectorLength);
    }

    for (auto const &[CI, SubVectorLength] : loadsToReplace) {
        lReplaceMaskedLoad(builder, CI, SubVectorLength);
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
