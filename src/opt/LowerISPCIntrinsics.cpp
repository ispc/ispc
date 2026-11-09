/*
  Copyright (c) 2024-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "LowerISPCIntrinsics.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LowerAtomic.h>

#include <numeric>

namespace ispc {

static llvm::Constant *lGetSequentialMask(llvm::IRBuilder<> &builder, unsigned N) {
    std::vector<unsigned> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    return llvm::ConstantDataVector::get(builder.getContext(), indices);
}

static unsigned lGetVecNumElements(llvm::Value *V) {
    auto *vecType = llvm::dyn_cast<llvm::VectorType>(V->getType());
    Assert(vecType);
    return vecType->getElementCount().getKnownMinValue();
}

static llvm::MDNode *lNonTemporalMetadata(llvm::LLVMContext &ctx) {
    return llvm::MDNode::get(ctx,
                             llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)));
}

static llvm::AtomicOrdering lSetMemoryOrdering(const std::string &str) {
    if (str == "unordered") {
        return llvm::AtomicOrdering::Unordered;
    } else if (str == "monotonic") {
        return llvm::AtomicOrdering::Monotonic;
    } else if (str == "acquire") {
        return llvm::AtomicOrdering::Acquire;
    } else if (str == "release") {
        return llvm::AtomicOrdering::Release;
    } else if (str == "acq_rel") {
        return llvm::AtomicOrdering::AcquireRelease;
    } else if (str == "seq_cst") {
        return llvm::AtomicOrdering::SequentiallyConsistent;
    }
    return llvm::AtomicOrdering::NotAtomic;
}

static llvm::Value *lTruncVectorToi1(llvm::IRBuilder<> &builder, llvm::Value *V) {
    llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(V->getType());
    Assert(VT);
    // check if the vector element type is not i1
    llvm::Type *ET = VT->getElementType();
    if (!ET->isIntegerTy(1)) {
        // truncate vector of i32/i16/i8 to vector of i1
        llvm::Type *i1 = llvm::IntegerType::get(builder.getContext(), 1);
        llvm::Type *newVT = llvm::VectorType::get(i1, lGetVecNumElements(V), false);
        V = builder.CreateTrunc(V, newVT);
    }
    return V;
}

static llvm::Value *lCreateSelect(llvm::IRBuilder<> &builder, llvm::Value *C, llvm::Value *T, llvm::Value *F) {
    // when C is not a vector of i1, we need to truncate it to i1
    // This is ugly hack due to the fact that varying bool is represented as
    // vector of i32/i16/i8 for some targets
    llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(C->getType());
    if (VT) {
        C = lTruncVectorToi1(builder, C);
    }

    return builder.CreateSelect(C, T, F);
}

static llvm::Value *lLowerConcatIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 2);

    llvm::Value *V0 = CI->getArgOperand(0);
    llvm::Value *V1 = CI->getArgOperand(1);

    auto N = lGetVecNumElements(V0);
    return builder.CreateShuffleVector(V0, V1, lGetSequentialMask(builder, 2 * N));
}

static llvm::Value *lLowerExtractIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    auto numArgs = CI->arg_size();

    if (numArgs == 2) {
        llvm::Value *V = CI->getArgOperand(0);
        llvm::Value *I = CI->getArgOperand(1);

        return builder.CreateExtractElement(V, I);
    }
    if (numArgs == 3) {
        llvm::Value *V0 = CI->getArgOperand(0);
        llvm::Value *V1 = CI->getArgOperand(1);
        llvm::Value *I = CI->getArgOperand(2);

        auto N = lGetVecNumElements(V0);
        llvm::Value *V = builder.CreateShuffleVector(V0, V1, lGetSequentialMask(builder, 2 * N));

        return builder.CreateExtractElement(V, I);
    }
    Assert(false);
    return nullptr;
}

static llvm::Value *lLowerInsertIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    llvm::Value *V = CI->getArgOperand(0);
    llvm::Value *I = CI->getArgOperand(1);
    llvm::Value *E = CI->getArgOperand(2);

    return builder.CreateInsertElement(V, E, I);
}

static llvm::Value *lLowerBitcastIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 2);

    llvm::Value *V = CI->getArgOperand(0);
    llvm::Value *VT = CI->getArgOperand(1);

    return builder.CreateBitCast(V, VT->getType());
}

static llvm::Value *lLowerBlendStore(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    llvm::Value *V = CI->getArgOperand(0);
    llvm::Value *P = CI->getArgOperand(1);
    llvm::Value *M = CI->getArgOperand(2);

    llvm::Type *T = V->getType();
    llvm::LoadInst *LI = builder.CreateLoad(T, P);
    llvm::Value *selected = lCreateSelect(builder, M, V, LI);
    llvm::StoreInst *SI = builder.CreateStore(selected, P);

    return SI;
}

static llvm::Value *lLowerStreamStoreIntrinsic(llvm::CallInst *CI) {
    // generate store with !nontemporal metadata attached
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 2);

    llvm::Value *P = CI->getArgOperand(0);
    llvm::Value *V = CI->getArgOperand(1);

    llvm::StoreInst *SI = builder.CreateStore(V, P);
    SI->setMetadata("nontemporal", lNonTemporalMetadata(CI->getContext()));

    return SI;
}

static llvm::Value *lLowerStreamLoadIntrinsic(llvm::CallInst *CI) {
    // generate load with !nontemporal metadata attached
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 2);

    llvm::Value *P = CI->getArgOperand(0);
    llvm::Type *T = CI->getArgOperand(1)->getType();

    llvm::LoadInst *LI = builder.CreateLoad(T, P);
    LI->setMetadata("nontemporal", lNonTemporalMetadata(CI->getContext()));

    return LI;
}

static llvm::Value *lLowerAtomicRMWIntrinsic(llvm::CallInst *CI) {
    // generate atomicrmw instruction fetching op and ordering from intrinsic name
    // llvm.ispc.atomicrmw.<op>.<memoryOrdering>.<type>
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 2);

    llvm::Value *P = CI->getArgOperand(0);
    llvm::Value *V = CI->getArgOperand(1);

    llvm::AtomicRMWInst::BinOp op = llvm::AtomicRMWInst::BinOp::BAD_BINOP;
    std::string opName = CI->getCalledFunction()->getName().str();
    opName = opName.substr(0, opName.find_last_of('.'));
    std::string memoryOrdering = opName.substr(opName.find_last_of('.') + 1);
    opName = opName.substr(0, opName.find_last_of('.'));
    opName = opName.substr(opName.find_last_of('.') + 1);
    if (opName == "xchg") {
        op = llvm::AtomicRMWInst::BinOp::Xchg;
    } else if (opName == "add") {
        op = llvm::AtomicRMWInst::BinOp::Add;
    } else if (opName == "sub") {
        op = llvm::AtomicRMWInst::BinOp::Sub;
    } else if (opName == "and") {
        op = llvm::AtomicRMWInst::BinOp::And;
    } else if (opName == "nand") {
        op = llvm::AtomicRMWInst::BinOp::Nand;
    } else if (opName == "or") {
        op = llvm::AtomicRMWInst::BinOp::Or;
    } else if (opName == "xor") {
        op = llvm::AtomicRMWInst::BinOp::Xor;
    } else if (opName == "max") {
        op = llvm::AtomicRMWInst::BinOp::Max;
    } else if (opName == "min") {
        op = llvm::AtomicRMWInst::BinOp::Min;
    } else if (opName == "umax") {
        op = llvm::AtomicRMWInst::BinOp::UMax;
    } else if (opName == "umin") {
        op = llvm::AtomicRMWInst::BinOp::UMin;
    } else if (opName == "fadd") {
        op = llvm::AtomicRMWInst::BinOp::FAdd;
    } else if (opName == "fsub") {
        op = llvm::AtomicRMWInst::BinOp::FSub;
    } else if (opName == "fmax") {
        op = llvm::AtomicRMWInst::BinOp::FMax;
    } else if (opName == "fmin") {
        op = llvm::AtomicRMWInst::BinOp::FMin;
    }
    Assert(op != llvm::AtomicRMWInst::BinOp::BAD_BINOP);

    llvm::AtomicOrdering ordering = lSetMemoryOrdering(memoryOrdering);
    Assert(ordering != llvm::AtomicOrdering::NotAtomic);

    return builder.CreateAtomicRMW(op, P, V, llvm::MaybeAlign(), ordering);
}

// Expands an atomicrmw of a vector value to a loop of cmpxchg of an integer of
// the same size. LLVM backends lower an atomicrmw of a vector to such a loop
// only if the vector is wider than their native atomic operations, and fail
// to select it otherwise (for example <2 x i32> on x86 and on AArch64 with
// LSE), so it's expanded for all vectors here.
static void lExpandVectorAtomicRMW(llvm::AtomicRMWInst *AI) {
    llvm::BasicBlock *BB = AI->getParent();
    llvm::Function *F = BB->getParent();
    llvm::LLVMContext &ctx = F->getContext();
    llvm::Type *valueType = AI->getType();
    llvm::Type *intType =
        llvm::IntegerType::get(ctx, static_cast<unsigned>(valueType->getPrimitiveSizeInBits().getFixedValue()));
    llvm::Value *P = AI->getPointerOperand();
    llvm::AtomicOrdering ordering = AI->getOrdering();

    llvm::BasicBlock *exitBB = BB->splitBasicBlock(AI, "atomicrmw.end");
    llvm::BasicBlock *loopBB = llvm::BasicBlock::Create(ctx, "atomicrmw.start", F, exitBB);
    // splitBasicBlock() terminates BB with a branch to exitBB, branch to the
    // loop instead.
    BB->getTerminator()->eraseFromParent();
    // The initial value is loaded atomically, as the location may be modified
    // concurrently, which makes a non-atomic load return undef.
    llvm::IRBuilder<> builder(BB);
    builder.SetCurrentDebugLocation(AI->getDebugLoc());
    llvm::LoadInst *initial = builder.CreateAlignedLoad(intType, P, AI->getAlign());
    initial->setAtomic(llvm::AtomicOrdering::Monotonic, AI->getSyncScopeID());
    llvm::Value *initialValue = builder.CreateBitCast(initial, valueType);
    builder.CreateBr(loopBB);

    builder.SetInsertPoint(loopBB);
    llvm::PHINode *loaded = builder.CreatePHI(valueType, 2, "loaded");
    loaded->addIncoming(initialValue, BB);
    llvm::Value *newValue = llvm::buildAtomicRMWValue(AI->getOperation(), builder, loaded, AI->getValOperand());
    // The bitcasts are created in separate statements, as the evaluation order
    // of function arguments is unspecified and the order of the instructions
    // would depend on the host compiler.
    llvm::Value *cmpInt = builder.CreateBitCast(loaded, intType);
    llvm::Value *newInt = builder.CreateBitCast(newValue, intType);
    llvm::Value *pair = builder.CreateAtomicCmpXchg(P, cmpInt, newInt, AI->getAlign(), ordering,
                                                    llvm::AtomicCmpXchgInst::getStrongestFailureOrdering(ordering),
                                                    AI->getSyncScopeID());
    llvm::Value *success = builder.CreateExtractValue(pair, 1, "success");
    llvm::Value *newLoaded = builder.CreateBitCast(builder.CreateExtractValue(pair, 0), valueType, "newloaded");
    loaded->addIncoming(newLoaded, loopBB);
    builder.CreateCondBr(success, exitBB, loopBB);

    AI->replaceAllUsesWith(newLoaded);
    AI->eraseFromParent();
}

static llvm::Value *lLowerCmpXchgIntrinsic(llvm::CallInst *CI) {
    // generate cmpxchg instruction fetching success ordering and failure
    // ordering from intrinsic name
    // llvm.ispc.cmpxchg.<successOrdering>.<failureOrdering>.<type>
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    llvm::Value *P = CI->getArgOperand(0);
    llvm::Value *C = CI->getArgOperand(1);
    llvm::Value *N = CI->getArgOperand(2);

    std::string prefix = CI->getCalledFunction()->getName().str();
    prefix = prefix.substr(0, prefix.find_last_of('.'));
    std::string failureOrdering = prefix.substr(prefix.find_last_of('.') + 1);
    prefix = prefix.substr(0, prefix.find_last_of('.'));
    std::string successOrdering = prefix.substr(prefix.find_last_of('.') + 1);

    llvm::AtomicOrdering SO = lSetMemoryOrdering(successOrdering);
    llvm::AtomicOrdering FO = lSetMemoryOrdering(failureOrdering);
    Assert(SO != llvm::AtomicOrdering::NotAtomic && FO != llvm::AtomicOrdering::NotAtomic);

    // cmpxchg returns a struct with two values, we only need the first one
    llvm::Value *result = builder.CreateAtomicCmpXchg(P, C, N, llvm::MaybeAlign(), SO, FO);
    return builder.CreateExtractValue(result, 0);
}

static llvm::Value *lLowerSelectIntrinsic(llvm::CallInst *CI) {
    // generate select instruction
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    llvm::Value *C = CI->getArgOperand(0);
    llvm::Value *T = CI->getArgOperand(1);
    llvm::Value *F = CI->getArgOperand(2);

    return lCreateSelect(builder, C, T, F);
}

static llvm::Value *lLowerFenceIntrinsic(llvm::CallInst *CI) {
    // generate fence instruction
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 0);

    std::string ordering = CI->getCalledFunction()->getName().str();
    ordering = ordering.substr(ordering.find_last_of('.') + 1);

    llvm::AtomicOrdering AO = lSetMemoryOrdering(ordering);
    Assert(AO != llvm::AtomicOrdering::NotAtomic);

    return builder.CreateFence(AO);
}

static llvm::Value *lLowerPackMaskIntrinsic(llvm::CallInst *CI) {
    // generate bitcast from <WIDTH x i1> to i`WIDTH if mask type is i1
    // otherwise truncate the mask from <WIDTH x i32|i16|i8> to <WIDTH x i1> before
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 1);

    llvm::Value *V = CI->getArgOperand(0);

    llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(V->getType());
    Assert(VT);
    V = lTruncVectorToi1(builder, V);

    // get type with the same width as target width
    llvm::Type *newVT = llvm::IntegerType::get(builder.getContext(), lGetVecNumElements(V));
    llvm::Value *packed = builder.CreateBitCast(V, newVT);

    // zero extend to i64
    return builder.CreateZExt(packed, llvm::Type::getInt64Ty(builder.getContext()));
}

static bool lRunOnBasicBlock(llvm::BasicBlock &BB, std::vector<llvm::AtomicRMWInst *> &vectorAtomics) {
    bool modified = false;
    for (llvm::BasicBlock::iterator iter = BB.begin(), e = BB.end(); iter != e;) {
        if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(&*(iter++))) {
            llvm::Function *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName().starts_with("llvm.ispc.")) {
                llvm::Value *D = nullptr;
                if (Callee->getName().starts_with("llvm.ispc.concat.")) {
                    D = lLowerConcatIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.extract.")) {
                    D = lLowerExtractIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.insert.")) {
                    D = lLowerInsertIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.bitcast.")) {
                    D = lLowerBitcastIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.blend_store.")) {
                    D = lLowerBlendStore(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.stream_store.")) {
                    D = lLowerStreamStoreIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.stream_load.")) {
                    D = lLowerStreamLoadIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.atomicrmw.")) {
                    D = lLowerAtomicRMWIntrinsic(CI);
                    if (D->getType()->isVectorTy()) {
                        vectorAtomics.push_back(llvm::cast<llvm::AtomicRMWInst>(D));
                    }
                } else if (Callee->getName().starts_with("llvm.ispc.cmpxchg.")) {
                    D = lLowerCmpXchgIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.select.")) {
                    D = lLowerSelectIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.fence.")) {
                    D = lLowerFenceIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.packmask.")) {
                    D = lLowerPackMaskIntrinsic(CI);
                }

                if (D) {
                    CI->replaceAllUsesWith(D);
                    CI->eraseFromParent();
                    modified = true;
                }
            }
        }
    }
    return modified;
}

llvm::PreservedAnalyses LowerISPCIntrinsicsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("LowerISPCIntrinsicsPass::run", F.getName());
    bool modified = false;
    std::vector<llvm::AtomicRMWInst *> vectorAtomics;
    for (llvm::BasicBlock &BB : F) {
        modified |= lRunOnBasicBlock(BB, vectorAtomics);
    }
    // The expansion splits basic blocks, so it's done after the loop above.
    for (llvm::AtomicRMWInst *AI : vectorAtomics) {
        lExpandVectorAtomicRMW(AI);
    }

    if (!modified) {
        return llvm::PreservedAnalyses::all();
    }
    llvm::PreservedAnalyses PA;
    if (vectorAtomics.empty()) {
        PA.preserveSet<llvm::CFGAnalyses>();
    }
    return PA;
}

} // namespace ispc
