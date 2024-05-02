/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "PeepholePass.h"
#include "builtins-decl.h"

namespace ispc {

using namespace llvm::PatternMatch;

template <typename Op_t, unsigned Opcode> struct CastClassTypes_match {
    Op_t Op;
    const llvm::Type *fromType, *toType;

    CastClassTypes_match(const Op_t &OpMatch, const llvm::Type *f, const llvm::Type *t)
        : Op(OpMatch), fromType(f), toType(t) {}

    template <typename OpTy> bool match(OpTy *V) {
        if (llvm::Operator *O = llvm::dyn_cast<llvm::Operator>(V))
            return (O->getOpcode() == Opcode && Op.match(O->getOperand(0)) && O->getType() == toType &&
                    O->getOperand(0)->getType() == fromType);
        return false;
    }
};

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::SExt> m_SExt8To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::SExt>(Op, LLVMTypes::Int8VectorType,
                                                               LLVMTypes::Int16VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::ZExt> m_ZExt8To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::ZExt>(Op, LLVMTypes::Int8VectorType,
                                                               LLVMTypes::Int16VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::Trunc> m_Trunc16To8(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::Trunc>(Op, LLVMTypes::Int16VectorType,
                                                                LLVMTypes::Int8VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::SExt> m_SExt16To32(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::SExt>(Op, LLVMTypes::Int16VectorType,
                                                               LLVMTypes::Int32VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::ZExt> m_ZExt16To32(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::ZExt>(Op, LLVMTypes::Int16VectorType,
                                                               LLVMTypes::Int32VectorType);
}

template <typename OpTy> inline CastClassTypes_match<OpTy, llvm::Instruction::Trunc> m_Trunc32To16(const OpTy &Op) {
    return CastClassTypes_match<OpTy, llvm::Instruction::Trunc>(Op, LLVMTypes::Int32VectorType,
                                                                LLVMTypes::Int16VectorType);
}

template <typename Op_t> struct UDiv2_match {
    Op_t Op;

    UDiv2_match(const Op_t &OpMatch) : Op(OpMatch) {}

    template <typename OpTy> bool match(OpTy *V) {
        llvm::BinaryOperator *bop;
        llvm::ConstantDataVector *cdv;
        if ((bop = llvm::dyn_cast<llvm::BinaryOperator>(V)) &&
            (cdv = llvm::dyn_cast<llvm::ConstantDataVector>(bop->getOperand(1))) && cdv->getSplatValue() != nullptr) {
            const llvm::APInt &apInt = cdv->getUniqueInteger();

            switch (bop->getOpcode()) {
            case llvm::Instruction::UDiv:
                // divide by 2
                return (apInt.isIntN(2) && Op.match(bop->getOperand(0)));
            case llvm::Instruction::LShr:
                // shift left by 1
                return (apInt.isIntN(1) && Op.match(bop->getOperand(0)));
            default:
                return false;
            }
        }
        return false;
    }
};

template <typename V> inline UDiv2_match<V> m_UDiv2(const V &v) { return UDiv2_match<V>(v); }

template <typename Op_t> struct SDiv2_match {
    Op_t Op;

    SDiv2_match(const Op_t &OpMatch) : Op(OpMatch) {}

    template <typename OpTy> bool match(OpTy *V) {
        llvm::BinaryOperator *bop;
        llvm::ConstantDataVector *cdv;
        if ((bop = llvm::dyn_cast<llvm::BinaryOperator>(V)) &&
            (cdv = llvm::dyn_cast<llvm::ConstantDataVector>(bop->getOperand(1))) && cdv->getSplatValue() != nullptr) {
            const llvm::APInt &apInt = cdv->getUniqueInteger();

            switch (bop->getOpcode()) {
            case llvm::Instruction::SDiv:
                // divide by 2
                return (apInt.isIntN(2) && Op.match(bop->getOperand(0)));
            case llvm::Instruction::AShr:
                // shift left by 1
                return (apInt.isIntN(1) && Op.match(bop->getOperand(0)));
            default:
                return false;
            }
        }
        return false;
    }
};

template <typename V> inline SDiv2_match<V> m_SDiv2(const V &v) { return SDiv2_match<V>(v); }

// Returns true if the given function has a call to an intrinsic function
// in its definition.
static bool lHasIntrinsicInDefinition(llvm::Function *func) {
    llvm::Function::iterator bbiter = func->begin();
    for (; bbiter != func->end(); ++bbiter) {
        for (llvm::BasicBlock::iterator institer = bbiter->begin(); institer != bbiter->end(); ++institer) {
            if (llvm::isa<llvm::IntrinsicInst>(institer))
                return true;
        }
    }
    return false;
}

static llvm::Instruction *lGetBinaryIntrinsic(llvm::Module *M, const char *name, llvm::Value *opa, llvm::Value *opb) {
    llvm::Function *func = M->getFunction(name);
    Assert(func != nullptr);

    // TODO: does it do something on, e.g., avx2-i32x4 target at all?
    //
    // Make sure that the definition of the llvm::Function has a call to an
    // intrinsic function in its instructions; otherwise we will generate
    // infinite loops where we "helpfully" turn the default implementations
    // of target builtins like __avg_up_uint8 that are implemented with plain
    // arithmetic ops into recursive calls to themselves.
    if (lHasIntrinsicInDefinition(func))
        return LLVMCallInst(func, opa, opb, name);
    else
        return nullptr;
}

//////////////////////////////////////////////////

static llvm::Instruction *lMatchAvgUpUInt8(llvm::Instruction *inst) {
    // (unsigned int8)(((unsigned int16)a + (unsigned int16)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst, m_Trunc16To8(m_UDiv2(m_CombineOr(
                        m_CombineOr(m_Add(m_ZExt8To16(m_Value(opa)), m_Add(m_ZExt8To16(m_Value(opb)), m_APInt(delta))),
                                    m_Add(m_Add(m_ZExt8To16(m_Value(opa)), m_APInt(delta)), m_ZExt8To16(m_Value(opb)))),
                        m_Add(m_Add(m_ZExt8To16(m_Value(opa)), m_ZExt8To16(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return nullptr;

        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_up_uint8, opa, opb);
    }
    return nullptr;
}

static llvm::Instruction *lMatchAvgDownUInt8(llvm::Instruction *inst) {
    // (unsigned int8)(((unsigned int16)a + (unsigned int16)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc16To8(m_UDiv2(m_Add(m_ZExt8To16(m_Value(opa)), m_ZExt8To16(m_Value(opb))))))) {
        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_down_uint8, opa, opb);
    }
    return nullptr;
}

static llvm::Instruction *lMatchAvgUpUInt16(llvm::Instruction *inst) {
    // (unsigned int16)(((unsigned int32)a + (unsigned int32)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst,
              m_Trunc32To16(m_UDiv2(m_CombineOr(
                  m_CombineOr(m_Add(m_ZExt16To32(m_Value(opa)), m_Add(m_ZExt16To32(m_Value(opb)), m_APInt(delta))),
                              m_Add(m_Add(m_ZExt16To32(m_Value(opa)), m_APInt(delta)), m_ZExt16To32(m_Value(opb)))),
                  m_Add(m_Add(m_ZExt16To32(m_Value(opa)), m_ZExt16To32(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return nullptr;

        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_up_uint16, opa, opb);
    }
    return nullptr;
}

static llvm::Instruction *lMatchAvgDownUInt16(llvm::Instruction *inst) {
    // (unsigned int16)(((unsigned int32)a + (unsigned int32)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc32To16(m_UDiv2(m_Add(m_ZExt16To32(m_Value(opa)), m_ZExt16To32(m_Value(opb))))))) {
        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_down_uint16, opa, opb);
    }
    return nullptr;
}

static llvm::Instruction *lMatchAvgUpInt8(llvm::Instruction *inst) {
    // (int8)(((int16)a + (int16)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst, m_Trunc16To8(m_SDiv2(m_CombineOr(
                        m_CombineOr(m_Add(m_SExt8To16(m_Value(opa)), m_Add(m_SExt8To16(m_Value(opb)), m_APInt(delta))),
                                    m_Add(m_Add(m_SExt8To16(m_Value(opa)), m_APInt(delta)), m_SExt8To16(m_Value(opb)))),
                        m_Add(m_Add(m_SExt8To16(m_Value(opa)), m_SExt8To16(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return nullptr;

        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_up_int8, opa, opb);
    }
    return nullptr;
}

static llvm::Instruction *lMatchAvgDownInt8(llvm::Instruction *inst) {
    // (int8)(((int16)a + (int16)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc16To8(m_SDiv2(m_Add(m_SExt8To16(m_Value(opa)), m_SExt8To16(m_Value(opb))))))) {
        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_down_int8, opa, opb);
    }
    return nullptr;
}

static llvm::Instruction *lMatchAvgUpInt16(llvm::Instruction *inst) {
    // (int16)(((int32)a + (int32)b + 1)/2)
    llvm::Value *opa, *opb;
    const llvm::APInt *delta;
    if (match(inst,
              m_Trunc32To16(m_SDiv2(m_CombineOr(
                  m_CombineOr(m_Add(m_SExt16To32(m_Value(opa)), m_Add(m_SExt16To32(m_Value(opb)), m_APInt(delta))),
                              m_Add(m_Add(m_SExt16To32(m_Value(opa)), m_APInt(delta)), m_SExt16To32(m_Value(opb)))),
                  m_Add(m_Add(m_SExt16To32(m_Value(opa)), m_SExt16To32(m_Value(opb))), m_APInt(delta))))))) {
        if (delta->isIntN(1) == false)
            return nullptr;

        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_up_int16, opa, opb);
    }
    return nullptr;
}

static llvm::Instruction *lMatchAvgDownInt16(llvm::Instruction *inst) {
    // (int16)(((int32)a + (int32)b)/2)
    llvm::Value *opa, *opb;
    if (match(inst, m_Trunc32To16(m_SDiv2(m_Add(m_SExt16To32(m_Value(opa)), m_SExt16To32(m_Value(opb))))))) {
        return lGetBinaryIntrinsic(inst->getModule(), builtin::__avg_down_int16, opa, opb);
    }
    return nullptr;
}

bool PeepholePass::matchAndReplace(llvm::BasicBlock &bb) {
    DEBUG_START_BB("PeepholePass");

    bool modifiedAny = false;

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::Instruction *inst = &*(iter++);

        llvm::Instruction *builtinCall = lMatchAvgUpUInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpUInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownUInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownUInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgUpInt16(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownInt8(inst);
        if (!builtinCall)
            builtinCall = lMatchAvgDownInt16(inst);
        if (builtinCall != nullptr) {
            llvm::ReplaceInstWithInst(inst, builtinCall);
            modifiedAny = true;
        }
    }

    DEBUG_END_BB("PeepholePass");

    return modifiedAny;
}

llvm::PreservedAnalyses PeepholePass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {

    llvm::TimeTraceScope FuncScope("PeepholePass::run", F.getName());
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
