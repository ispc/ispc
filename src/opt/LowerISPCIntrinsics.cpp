/*
  Copyright (c) 2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "LowerISPCIntrinsics.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Local.h>

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

static llvm::Value *lLowerConcatIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    llvm::Value *V0 = CI->getArgOperand(0);
    llvm::Value *V1 = CI->getArgOperand(1);

    auto N = lGetVecNumElements(V0);
    return builder.CreateShuffleVector(V0, V1, lGetSequentialMask(builder, 2 * N));
}

static llvm::Value *lLowerExtractIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    auto numArgs = CI->getNumOperands() - 1;

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
    return nullptr;
}

static llvm::Value *lLowerInserIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);

    llvm::Value *V = CI->getArgOperand(0);
    llvm::Value *I = CI->getArgOperand(1);
    llvm::Value *E = CI->getArgOperand(2);

    return builder.CreateInsertElement(V, E, I);
}

static llvm::Value *lLowerBitcastIntrinsic(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);

    llvm::Value *V = CI->getArgOperand(0);
    llvm::Value *VT = CI->getArgOperand(1);

    return builder.CreateBitCast(V, VT->getType());
}

static llvm::MDNode *lNonTemporalMetadata(llvm::LLVMContext &ctx) {
    return llvm::MDNode::get(ctx,
                             llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)));
}

static llvm::Value *lLowerStreamStoreIntrinsic(llvm::CallInst *CI) {
    // generate store with !nontemporal metadata attached
    llvm::IRBuilder<> builder(CI);

    llvm::Value *P = CI->getArgOperand(0);
    llvm::Value *V = CI->getArgOperand(1);

    llvm::StoreInst *SI = builder.CreateStore(V, P);
    SI->setMetadata("nontemporal", lNonTemporalMetadata(CI->getContext()));

    return SI;
}

static llvm::Value *lLowerStreamLoadIntrinsic(llvm::CallInst *CI) {
    // generate load with !nontemporal metadata attached
    llvm::IRBuilder<> builder(CI);

    llvm::Value *P = CI->getArgOperand(0);
    llvm::Type *T = CI->getArgOperand(1)->getType();

    llvm::LoadInst *LI = builder.CreateLoad(T, P);
    LI->setMetadata("nontemporal", lNonTemporalMetadata(CI->getContext()));

    return LI;
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

static llvm::Value *lLowerAtomicRMWIntrinsic(llvm::CallInst *CI) {
    // generate atomicrmw instruction fetching op and ordering from intrinsic name
    // llvm.ispc.atomicrmw.<op>.<memoryOrdering>.<type>
    llvm::IRBuilder<> builder(CI);

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

static llvm::Value *lLowerCmpXchgIntrinsic(llvm::CallInst *CI) {
    // generate cmpxchg instruction fetching success ordering and failure
    // ordering from intrinsic name
    // llvm.ispc.cmpxchg.<successOrdering>.<failureOrdering>.<type>
    llvm::IRBuilder<> builder(CI);

    llvm::Value *P = CI->getArgOperand(0);
    llvm::Value *C = CI->getArgOperand(1);
    llvm::Value *N = CI->getArgOperand(2);

    std::string prefix = CI->getCalledFunction()->getName().str();
    prefix = prefix.substr(0, prefix.find_last_of('.'));
    std::string successOrdering = prefix.substr(prefix.find_last_of('.') + 1);
    prefix = prefix.substr(0, prefix.find_last_of('.'));
    std::string failureOrdering = prefix.substr(prefix.find_last_of('.') + 1);

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

    llvm::Value *C = CI->getArgOperand(0);
    llvm::Value *T = CI->getArgOperand(1);
    llvm::Value *F = CI->getArgOperand(2);

    // when C is not a vector of i1, we need to truncate it to i1
    // This is ugly hack due to the fact that varying bool is represented as
    // vector of i32/i16/i8 for some targets
    llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(C->getType());
    if (VT) {
        // check if the vector element type is not i1
        llvm::Type *ET = VT->getElementType();
        if (!ET->isIntegerTy(1)) {
            // truncate vector of i32/i16/i8 to vector of i1
            llvm::Type *i1 = llvm::IntegerType::get(builder.getContext(), 1);
            llvm::Type *newVT = llvm::VectorType::get(i1, lGetVecNumElements(C), false);
            C = builder.CreateTrunc(C, newVT);
        }
    }

    return builder.CreateSelect(C, T, F);
}

static llvm::Value *lLowerFenceIntrinsic(llvm::CallInst *CI) {
    // generate fence instruction
    llvm::IRBuilder<> builder(CI);

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

    llvm::Value *V = CI->getArgOperand(0);
    llvm::VectorType *VT = llvm::dyn_cast<llvm::VectorType>(CI->getArgOperand(0)->getType());
    Assert(VT);

    // check if the vector element type is not i1
    llvm::Type *ET = VT->getElementType();
    if (!ET->isIntegerTy(1)) {
        // truncate vector of i32/i16/i8 to vector of i1
        llvm::Type *i1 = llvm::IntegerType::get(builder.getContext(), 1);
        llvm::Type *newVT = llvm::VectorType::get(i1, lGetVecNumElements(V), false);
        V = builder.CreateTrunc(V, newVT);
    }

    // get type with the same width as target width
    llvm::Type *newVT = llvm::IntegerType::get(builder.getContext(), lGetVecNumElements(V));
    llvm::Value *packed = builder.CreateBitCast(V, newVT);

    // zero extend to i64
    return builder.CreateZExt(packed, llvm::Type::getInt64Ty(builder.getContext()));
}

static bool lRunOnBasicBlock(llvm::BasicBlock &BB) {
    // TODO: add lit tests
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
                    D = lLowerInserIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.bitcast.")) {
                    D = lLowerBitcastIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.stream_store.")) {
                    D = lLowerStreamStoreIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.stream_load.")) {
                    D = lLowerStreamLoadIntrinsic(CI);
                } else if (Callee->getName().starts_with("llvm.ispc.atomicrmw.")) {
                    D = lLowerAtomicRMWIntrinsic(CI);
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
                }
            }
        }
    }
    return false;
}

llvm::PreservedAnalyses LowerISPCIntrinsicsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("LowerISPCIntrinsicsPass::run", F.getName());
    bool modified = false;
    for (llvm::BasicBlock &BB : F) {
        modified |= lRunOnBasicBlock(BB);
    }

    if (modified) {
        llvm::PreservedAnalyses PA;
        PA.preserveSet<llvm::CFGAnalyses>();
        return PA;
    } else {
        return llvm::PreservedAnalyses::all();
    }
}

} // namespace ispc
