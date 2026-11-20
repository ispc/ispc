/*
  Copyright (c) 2024-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ScalarizePass.h"
#include "builtins-decl.h"

#include "llvm/IR/IRBuilder.h"
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

// Recover a scalar byte offset from a splat OR a constant lane offset.
static llvm::Value *lScalarizeComputedByteOffset(llvm::Value *indices, uint64_t lane, llvm::IRBuilder<> &builder) {
    auto *op = llvm::dyn_cast<llvm::BinaryOperator>(indices);
    if (!op || op->getOpcode() != llvm::Instruction::Or) {
        return nullptr;
    }
    auto *offsets = llvm::dyn_cast<llvm::Constant>(op->getOperand(1));
    auto *splat = llvm::dyn_cast<llvm::ShuffleVectorInst>(op->getOperand(0));
    if (!offsets || !splat || !llvm::isa<llvm::PoisonValue>(splat->getOperand(1))) {
        return nullptr;
    }
    for (int element : splat->getShuffleMask()) {
        if (element != 0) {
            return nullptr;
        }
    }
    auto *insert = llvm::dyn_cast<llvm::InsertElementInst>(splat->getOperand(0));
    auto *index = insert ? llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2)) : nullptr;
    auto *offset = llvm::dyn_cast_or_null<llvm::ConstantInt>(offsets->getAggregateElement(unsigned(lane)));
    if (!index || !index->isZero() || !llvm::isa<llvm::PoisonValue>(insert->getOperand(0)) || !offset ||
        offset->isNegative()) {
        return nullptr;
    }
    return builder.CreateOr(insert->getOperand(1), offset);
}

// Recover a scalar address for an ordinary byte load from computed offsets.
static llvm::Value *lScalarizeByteGatherIndex(llvm::Instruction *inst, unsigned width) {
    auto *extract = llvm::dyn_cast<llvm::ExtractElementInst>(inst);
    if (!extract || !extract->hasOneUse() || !llvm::isa<llvm::ConstantInt>(extract->getIndexOperand())) {
        return nullptr;
    }
    auto *extend = llvm::dyn_cast<llvm::ZExtInst>(extract->getVectorOperand());
    auto *address = llvm::dyn_cast<llvm::GetElementPtrInst>(*extract->user_begin());
    if (!extend || !address || !address->getSourceElementType()->isIntegerTy(8) || !address->hasOneUse() ||
        !extract->getType()->isIntegerTy(64)) {
        return nullptr;
    }
    auto *indices = extend->getOperand(0);
    auto *byte = llvm::dyn_cast<llvm::LoadInst>(*address->user_begin());
    if (!byte || !byte->isSimple() || byte->getPointerOperand() != address || !byte->getType()->isIntegerTy(8) ||
        indices->getType() != llvm::FixedVectorType::get(llvm::Type::getInt32Ty(inst->getContext()), width)) {
        return nullptr;
    }

    auto *lane = llvm::cast<llvm::ConstantInt>(extract->getIndexOperand());
    if (lane->getValue().uge(width)) {
        return nullptr;
    }
    llvm::IRBuilder<> builder(extract);
    auto *index = lScalarizeComputedByteOffset(indices, lane->getZExtValue(), builder);
    if (!index) {
        return nullptr;
    }
    return builder.CreateZExt(index, extract->getType(), "", extend->hasNonNeg());
}

bool ScalarizePass::matchAndReplace(llvm::BasicBlock &bb) {
    DEBUG_START_BB("ScalarizePass");

    bool modifiedAny = false;

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::Instruction *inst = &*(iter++);

        llvm::Value *newValue = lMatchAndScalarize(inst);
        if (!newValue && g && g->target && g->target->getISA() == Target::NVL_AVX10_2 &&
            (g->target->getVectorWidth() == 8 || g->target->getVectorWidth() == 16)) {
            newValue = lScalarizeByteGatherIndex(inst, g->target->getVectorWidth());
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
