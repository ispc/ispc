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

#include "IntrinsicsOptPass.h"

namespace ispc {

char IntrinsicsOpt::ID = 0;

bool IntrinsicsOpt::optimizeIntrinsics(llvm::BasicBlock &bb) {
    DEBUG_START_BB("IntrinsicsOpt");

    // We can't initialize mask/blend function vector during pass initialization,
    // as they may be optimized out by the time the pass is invoked.

    // All of the mask instructions we may encounter.  Note that even if
    // compiling for AVX, we may still encounter the regular 4-wide SSE
    // MOVMSK instruction.
    if (llvm::Function *ssei8Movmsk =
            m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse2_pmovmskb_128))) {
        maskInstructions.push_back(ssei8Movmsk);
    }
    if (llvm::Function *sseFloatMovmsk =
            m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse_movmsk_ps))) {
        maskInstructions.push_back(sseFloatMovmsk);
    }
    if (llvm::Function *__movmsk = m->module->getFunction("__movmsk")) {
        maskInstructions.push_back(__movmsk);
    }
    if (llvm::Function *avxFloatMovmsk =
            m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_movmsk_ps_256))) {
        maskInstructions.push_back(avxFloatMovmsk);
    }

    // And all of the blend instructions
    blendInstructions.push_back(BlendInstruction(
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_sse41_blendvps)), 0xf, 0, 1, 2));
    blendInstructions.push_back(BlendInstruction(
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_blendv_ps_256)), 0xff, 0, 1, 2));

    llvm::Function *avxMaskedLoad32 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskload_ps_256));
    llvm::Function *avxMaskedLoad64 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskload_pd_256));
    llvm::Function *avxMaskedStore32 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskstore_ps_256));
    llvm::Function *avxMaskedStore64 =
        m->module->getFunction(llvm::Intrinsic::getName(llvm::Intrinsic::x86_avx_maskstore_pd_256));

    bool modifiedAny = false;
restart:
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e; ++iter) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*iter);
        if (callInst == NULL || callInst->getCalledFunction() == NULL)
            continue;

        BlendInstruction *blend = matchingBlendInstruction(callInst->getCalledFunction());
        if (blend != NULL) {
            llvm::Value *v[2] = {callInst->getArgOperand(blend->op0), callInst->getArgOperand(blend->op1)};
            llvm::Value *factor = callInst->getArgOperand(blend->opFactor);

            // If the values are the same, then no need to blend..
            if (v[0] == v[1]) {
                ReplaceInstWithValueWrapper(iter, v[0]);
                modifiedAny = true;
                goto restart;
            }

            // If one of the two is undefined, we're allowed to replace
            // with the value of the other.  (In other words, the only
            // valid case is that the blend factor ends up having a value
            // that only selects from the defined one of the two operands,
            // otherwise the result is undefined and any value is fine,
            // ergo the defined one is an acceptable result.)
            if (LLVMIsValueUndef(v[0])) {
                ReplaceInstWithValueWrapper(iter, v[1]);
                modifiedAny = true;
                goto restart;
            }
            if (LLVMIsValueUndef(v[1])) {
                ReplaceInstWithValueWrapper(iter, v[0]);
                modifiedAny = true;
                goto restart;
            }

            MaskStatus maskStatus = GetMaskStatusFromValue(factor);
            llvm::Value *value = NULL;
            if (maskStatus == MaskStatus::all_off) {
                // Mask all off -> replace with the first blend value
                value = v[0];
            } else if (maskStatus == MaskStatus::all_on) {
                // Mask all on -> replace with the second blend value
                value = v[1];
            }

            if (value != NULL) {
                ReplaceInstWithValueWrapper(iter, value);
                modifiedAny = true;
                goto restart;
            }
        } else if (matchesMaskInstruction(callInst->getCalledFunction())) {
            llvm::Value *factor = callInst->getArgOperand(0);
            uint64_t mask;
            if (GetMaskFromValue(factor, &mask) == true) {
                // If the vector-valued mask has a known value, replace it
                // with the corresponding integer mask from its elements
                // high bits.
                llvm::Value *value = (callInst->getType() == LLVMTypes::Int32Type) ? LLVMInt32(mask) : LLVMInt64(mask);
                ReplaceInstWithValueWrapper(iter, value);
                modifiedAny = true;
                goto restart;
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
                ReplaceInstWithValueWrapper(iter, undefValue);
                modifiedAny = true;
                goto restart;
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
                    false /* not volatile */, llvm::MaybeAlign(align).valueOrOne(), (llvm::Instruction *)NULL);
                LLVMCopyMetadata(loadInst, callInst);
                llvm::ReplaceInstWithInst(callInst, loadInst);
                modifiedAny = true;
                goto restart;
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
                goto restart;
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
                llvm::StoreInst *storeInst = new llvm::StoreInst(rvalue, castPtr, (llvm::Instruction *)NULL,
                                                                 llvm::MaybeAlign(align).valueOrOne());
                LLVMCopyMetadata(storeInst, callInst);
                llvm::ReplaceInstWithInst(callInst, storeInst);

                modifiedAny = true;
                goto restart;
            }
        }
    }

    DEBUG_END_BB("IntrinsicsOpt");

    return modifiedAny;
}

bool IntrinsicsOpt::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("IntrinsicsOpt::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= optimizeIntrinsics(BB);
    }
    return modifiedAny;
}

bool IntrinsicsOpt::matchesMaskInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < maskInstructions.size(); ++i) {
        if (maskInstructions[i].function != NULL && function == maskInstructions[i].function) {
            return true;
        }
    }
    return false;
}

IntrinsicsOpt::BlendInstruction *IntrinsicsOpt::matchingBlendInstruction(llvm::Function *function) {
    for (unsigned int i = 0; i < blendInstructions.size(); ++i) {
        if (blendInstructions[i].function != NULL && function == blendInstructions[i].function) {
            return &blendInstructions[i];
        }
    }
    return NULL;
}

llvm::Pass *CreateIntrinsicsOptPass() { return new IntrinsicsOpt; }

} // namespace ispc
