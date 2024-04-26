/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "IntrinsicsOptPass.h"
#include "builtins-decl.h"

namespace ispc {

bool IntrinsicsOpt::optimizeIntrinsics(llvm::BasicBlock &bb) {
    DEBUG_START_BB("IntrinsicsOpt");

    // We can't initialize mask/blend function vector during pass initialization,
    // as they may be optimized out by the time the pass is invoked.

    // All of the mask instructions we may encounter.  Note that even if
    // compiling for AVX, we may still encounter the regular 4-wide SSE
    // MOVMSK instruction.
    llvm::Module *M = bb.getModule();
    if (llvm::Function *ssei8Movmsk =
            M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse2_pmovmskb_128))) {
        maskInstructions.push_back(ssei8Movmsk);
    }
    if (llvm::Function *sseFloatMovmsk = M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse_movmsk_ps))) {
        maskInstructions.push_back(sseFloatMovmsk);
    }
    if (llvm::Function *movmsk = M->getFunction(builtin::__movmsk)) {
        maskInstructions.push_back(movmsk);
    }
    if (llvm::Function *avxFloatMovmsk =
            M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_movmsk_ps_256))) {
        maskInstructions.push_back(avxFloatMovmsk);
    }

    // And all of the blend instructions
    blendInstructions.push_back(
        BlendInstruction(M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse41_blendvps)), 0xf, 0, 1, 2));
    blendInstructions.push_back(BlendInstruction(
        M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_blendv_ps_256)), 0xff, 0, 1, 2));

    llvm::Function *avxMaskedLoad32 =
        M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskload_ps_256));
    llvm::Function *avxMaskedLoad64 =
        M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskload_pd_256));
    llvm::Function *avxMaskedStore32 =
        M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskstore_ps_256));
    llvm::Function *avxMaskedStore64 =
        M->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskstore_pd_256));

    bool modifiedAny = false;

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::BasicBlock::iterator curIter = iter++;
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*(curIter));
        if (callInst == nullptr || callInst->getCalledFunction() == nullptr)
            continue;

        BlendInstruction *blend = matchingBlendInstruction(callInst->getCalledFunction());
        if (blend != nullptr) {
            llvm::Value *v[2] = {callInst->getArgOperand(blend->op0), callInst->getArgOperand(blend->op1)};
            llvm::Value *factor = callInst->getArgOperand(blend->opFactor);

            // If the values are the same, then no need to blend..
            if (v[0] == v[1]) {
                ReplaceInstWithValueWrapper(curIter, v[0]);
                modifiedAny = true;
                continue;
            }

            // If one of the two is undefined, we're allowed to replace
            // with the value of the other.  (In other words, the only
            // valid case is that the blend factor ends up having a value
            // that only selects from the defined one of the two operands,
            // otherwise the result is undefined and any value is fine,
            // ergo the defined one is an acceptable result.)
            if (LLVMIsValueUndef(v[0])) {
                ReplaceInstWithValueWrapper(curIter, v[1]);
                modifiedAny = true;
                continue;
            }
            if (LLVMIsValueUndef(v[1])) {
                ReplaceInstWithValueWrapper(curIter, v[0]);
                modifiedAny = true;
                continue;
            }

            MaskStatus maskStatus = GetMaskStatusFromValue(factor);
            llvm::Value *value = nullptr;
            if (maskStatus == MaskStatus::all_off) {
                // Mask all off -> replace with the first blend value
                value = v[0];
            } else if (maskStatus == MaskStatus::all_on) {
                // Mask all on -> replace with the second blend value
                value = v[1];
            }

            if (value != nullptr) {
                ReplaceInstWithValueWrapper(curIter, value);
                modifiedAny = true;
                continue;
            }
        } else if (matchesMaskInstruction(callInst->getCalledFunction())) {
            llvm::Value *factor = callInst->getArgOperand(0);
            uint64_t mask;
            if (GetMaskFromValue(factor, &mask) == true) {
                // If the vector-valued mask has a known value, replace it
                // with the corresponding integer mask from its elements
                // high bits.
                llvm::Value *value = (callInst->getType() == LLVMTypes::Int32Type) ? LLVMInt32(mask) : LLVMInt64(mask);
                ReplaceInstWithValueWrapper(curIter, value);
                modifiedAny = true;
                continue;
            }
        } else if (callInst->getCalledFunction() == avxMaskedLoad32 ||
                   callInst->getCalledFunction() == avxMaskedLoad64) {
            llvm::Value *factor = callInst->getArgOperand(1);
            MaskStatus maskStatus = GetMaskStatusFromValue(factor);
            if (maskStatus == MaskStatus::all_off) {
                // nothing being loaded, replace with undef value
                llvm::Type *returnType = callInst->getType();
                Assert(llvm::isa<llvm::VectorType>(returnType));
                llvm::Value *undefValue = llvm::UndefValue::get(returnType);
                ReplaceInstWithValueWrapper(curIter, undefValue);
                modifiedAny = true;
                continue;
            } else if (maskStatus == MaskStatus::all_on) {
                // all lanes active; replace with a regular load
                llvm::Type *returnType = callInst->getType();
                Assert(llvm::isa<llvm::VectorType>(returnType));
                // cast the i8 * to the appropriate type
                llvm::Value *castPtr =
                    new llvm::BitCastInst(callInst->getArgOperand(0), llvm::PointerType::get(returnType, 0),
                                          llvm::Twine(callInst->getArgOperand(0)->getName()) + "_cast", callInst);
                LLVMCopyMetadata(castPtr, callInst);
                int align;
                if (g->opt.forceAlignedMemory)
                    align = g->target->getNativeVectorAlignment();
                else
                    align = callInst->getCalledFunction() == avxMaskedLoad32 ? 4 : 8;
                llvm::Instruction *loadInst = new llvm::LoadInst(
                    returnType, castPtr, llvm::Twine(callInst->getArgOperand(0)->getName()) + "_load",
                    false /* not volatile */, llvm::MaybeAlign(align).valueOrOne(), (llvm::Instruction *)nullptr);
                LLVMCopyMetadata(loadInst, callInst);
                llvm::ReplaceInstWithInst(callInst, loadInst);
                modifiedAny = true;
                continue;
            }
        } else if (callInst->getCalledFunction() == avxMaskedStore32 ||
                   callInst->getCalledFunction() == avxMaskedStore64) {
            // NOTE: mask is the 2nd parameter, not the 3rd one!!
            llvm::Value *factor = callInst->getArgOperand(1);
            MaskStatus maskStatus = GetMaskStatusFromValue(factor);
            if (maskStatus == MaskStatus::all_off) {
                // nothing actually being stored, just remove the inst
                callInst->eraseFromParent();
                modifiedAny = true;
                continue;
            } else if (maskStatus == MaskStatus::all_on) {
                // all lanes storing, so replace with a regular store
                llvm::Value *rvalue = callInst->getArgOperand(2);
                llvm::Type *storeType = rvalue->getType();
                llvm::Value *castPtr =
                    new llvm::BitCastInst(callInst->getArgOperand(0), llvm::PointerType::get(storeType, 0),
                                          llvm::Twine(callInst->getArgOperand(0)->getName()) + "_ptrcast", callInst);
                LLVMCopyMetadata(castPtr, callInst);

                int align;
                if (g->opt.forceAlignedMemory)
                    align = g->target->getNativeVectorAlignment();
                else
                    align = callInst->getCalledFunction() == avxMaskedStore32 ? 4 : 8;
                llvm::StoreInst *storeInst = new llvm::StoreInst(rvalue, castPtr, (llvm::Instruction *)nullptr,
                                                                 llvm::MaybeAlign(align).valueOrOne());
                LLVMCopyMetadata(storeInst, callInst);
                llvm::ReplaceInstWithInst(callInst, storeInst);

                modifiedAny = true;
                continue;
            }
        }
    }

    DEBUG_END_BB("IntrinsicsOpt");

    return modifiedAny;
}

llvm::PreservedAnalyses IntrinsicsOpt::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {

    llvm::TimeTraceScope FuncScope("IntrinsicsOpt::run", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= optimizeIntrinsics(BB);
    }
    if (!modifiedAny) {
        // No changes, all analyses are preserved.
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

bool IntrinsicsOpt::matchesMaskInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < maskInstructions.size(); ++i) {
        if (maskInstructions[i].function != nullptr && function == maskInstructions[i].function) {
            return true;
        }
    }
    return false;
}

IntrinsicsOpt::BlendInstruction *IntrinsicsOpt::matchingBlendInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < blendInstructions.size(); ++i) {
        if (blendInstructions[i].function != nullptr && function == blendInstructions[i].function) {
            return &blendInstructions[i];
        }
    }
    return nullptr;
}

} // namespace ispc
