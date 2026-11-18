/*
  Copyright (c) 2024-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ScalarizePass.h"
#include "builtins-decl.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils/Local.h"

namespace ispc {

using namespace llvm::PatternMatch;

static llvm::Value *lMatchAndScalarize(llvm::Instruction *inst) {
    llvm::Value *Vec = nullptr, *OutZeroMask = nullptr;
    llvm::BinaryOperator *BO = nullptr;
    if (!match(inst, m_Shuffle(m_BinOp(BO), m_Poison(), m_ZeroMask())) &&
        !match(inst, m_Shuffle(m_BinOp(BO), m_Undef(), m_ZeroMask()))) {
        return nullptr;
    }

    llvm::ShuffleVectorInst *SVI = llvm::cast<llvm::ShuffleVectorInst>(inst);
    OutZeroMask = SVI->getShuffleMaskForBitcode();

    llvm::BinaryOperator::BinaryOps Opc = BO->getOpcode();
    llvm::Value *Op1 = nullptr, *Op2 = nullptr;
    if (match(BO, m_BinOp(m_InsertElt(m_Undef(), m_Value(Op1), m_ZeroInt()), m_Value(Vec))) ||
        match(BO, m_BinOp(m_Value(Vec), m_InsertElt(m_Undef(), m_Value(Op2), m_ZeroInt())))) {

        if (llvm::ConstantVector *CV = llvm::dyn_cast<llvm::ConstantVector>(Vec)) {
            unsigned N = CV->getType()->getNumElements();
            bool isTailPoison = true;
            unsigned EltIdx = 0;

            for (unsigned i = 1; i < N; i++) {
                auto E = CV->getAggregateElement(i);
                if (!llvm::isa<llvm::PoisonValue>(E) && !llvm::isa<llvm::UndefValue>(E)) {
                    isTailPoison = false;
                    break;
                }
            }

            if (isTailPoison) {
                (!Op1 ? Op1 : Op2) = CV->getAggregateElement(EltIdx);
                Assert(Op1 && Op2);

                llvm::Type *VecType = Vec->getType();
                llvm::Value *PV = llvm::PoisonValue::get(VecType);
                llvm::IRBuilder<> Builder(inst);

                llvm::Value *ScalarBinOp = Builder.CreateBinOp(Opc, Op1, Op2);
                llvm::Value *newVec = Builder.CreateInsertElement(PV, ScalarBinOp, (uint64_t)0);
                return Builder.CreateShuffleVector(newVec, PV, OutZeroMask);
            }
        }
    }
    return nullptr;
}

static llvm::Value *lScalarizeInt64Clamp(llvm::Instruction *inst, unsigned gangWidth) {
    auto *minimum = llvm::dyn_cast<llvm::IntrinsicInst>(inst);
    auto *type = llvm::dyn_cast<llvm::FixedVectorType>(inst->getType());
    if (!minimum || !type || type->getNumElements() != gangWidth || !type->getElementType()->isIntegerTy(64)) {
        return nullptr;
    }
    const auto minID = minimum->getIntrinsicID();
    if (minID != llvm::Intrinsic::smin && minID != llvm::Intrinsic::umin) {
        return nullptr;
    }
    const auto maxID = minID == llvm::Intrinsic::smin ? llvm::Intrinsic::smax : llvm::Intrinsic::umax;
    for (unsigned i = 0; i < 2; ++i) {
        auto *maximum = llvm::dyn_cast<llvm::IntrinsicInst>(minimum->getArgOperand(i));
        if (!maximum || maximum->getIntrinsicID() != maxID || !maximum->hasOneUse()) {
            continue;
        }

        // On SSE2, scalar min/max pairs cost less than two emulated packed
        // 64-bit comparisons and their blends. Retain the min(max()) order,
        // including when the lower bound exceeds the upper bound.
        llvm::IRBuilder<> builder(inst);
        llvm::Value *result = llvm::PoisonValue::get(type);
        for (unsigned lane = 0; lane < type->getNumElements(); ++lane) {
            llvm::Value *value = builder.CreateExtractElement(maximum->getArgOperand(0), lane);
            llvm::Value *low = builder.CreateExtractElement(maximum->getArgOperand(1), lane);
            llvm::Value *high = builder.CreateExtractElement(minimum->getArgOperand(1 - i), lane);
            llvm::Value *lower = builder.CreateBinaryIntrinsic(maxID, value, low);
            llvm::Value *clamped = builder.CreateBinaryIntrinsic(minID, lower, high);
            result = builder.CreateInsertElement(result, clamped, lane);
        }
        return result;
    }
    return nullptr;
}

bool ScalarizePass::matchAndReplace(llvm::BasicBlock &bb) {
    DEBUG_START_BB("ScalarizePass");

    bool modifiedAny = false;
    const bool isSSE2 = g->target && (g->target->getISPCTarget() == ISPCTarget::sse2_i32x4 ||
                                      g->target->getISPCTarget() == ISPCTarget::sse2_i32x8);

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::Instruction *inst = &*(iter++);

        llvm::Value *newValue = lMatchAndScalarize(inst);
        if (newValue == nullptr && isSSE2) {
            newValue = lScalarizeInt64Clamp(inst, g->target->getVectorWidth());
        }
        if (newValue != nullptr) {
            inst->replaceAllUsesWith(newValue);
            llvm::RecursivelyDeleteTriviallyDeadInstructions(inst);
            modifiedAny = true;
        }
    }

    DEBUG_END_BB("ScalarizePass");

    return modifiedAny;
}

llvm::PreservedAnalyses ScalarizePass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("ScalarizePass::run", F.getName());

    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= matchAndReplace(BB);
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
