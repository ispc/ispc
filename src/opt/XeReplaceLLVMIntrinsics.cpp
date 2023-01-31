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

#include "XeReplaceLLVMIntrinsics.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {

char ReplaceLLVMIntrinsics::ID = 0;

bool ReplaceLLVMIntrinsics::replaceUnspportedIntrinsics(llvm::BasicBlock &bb) {
    DEBUG_START_BB("LLVM intrinsics replacement");
    std::vector<llvm::AllocaInst *> Allocas;

    bool modifiedAny = false;

restart:
    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            if (func == NULL || !func->isIntrinsic())
                continue;

            if (func->getName().equals("llvm.trap")) {
                llvm::Type *argTypes[] = {LLVMTypes::Int1VectorType, LLVMTypes::Int16VectorType};
                // Description of parameters for genx_raw_send_noresult can be found in target-genx.ll
                auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(
                    m->module, llvm::GenXIntrinsic::genx_raw_send_noresult, argTypes);
                llvm::SmallVector<llvm::Value *, 8> Args;
                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 0));
                Args.push_back(llvm::ConstantVector::getSplat(
                    llvm::ElementCount::get(static_cast<unsigned int>(g->target->getNativeVectorWidth()), false),
                    llvm::ConstantInt::getTrue(*g->ctx)));

                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 39));
                Args.push_back(llvm::ConstantInt::get(LLVMTypes::Int32Type, 33554448));
                llvm::Value *zeroMask = llvm::ConstantVector::getSplat(
                    llvm::ElementCount::get(static_cast<unsigned int>(g->target->getNativeVectorWidth()), false),
                    llvm::Constant::getNullValue(llvm::Type::getInt16Ty(*g->ctx)));
                Args.push_back(zeroMask);

                llvm::Instruction *newInst = llvm::CallInst::Create(Fn, Args, ci->getName());
                if (newInst != NULL) {
                    llvm::ReplaceInstWithInst(ci, newInst);
                    modifiedAny = true;
                    goto restart;
                }
            } else if (func->getName().equals("llvm.experimental.noalias.scope.decl")) {
                // These intrinsics are not supported by backend so remove them.
                ci->eraseFromParent();
                modifiedAny = true;
                goto restart;
            } else if (func->getName().contains("llvm.abs")) {
                // Replace llvm.asb with llvm.genx.aba.alternative
                Assert(ci->getOperand(0));
                llvm::Type *argType = ci->getOperand(0)->getType();

                llvm::Type *Tys[2];
                Tys[0] = func->getReturnType(); // return type
                Tys[1] = argType;               // value type

                llvm::GenXIntrinsic::ID xeAbsID =
                    argType->isIntOrIntVectorTy() ? llvm::GenXIntrinsic::genx_absi : llvm::GenXIntrinsic::genx_absf;
                auto Fn = llvm::GenXIntrinsic::getGenXDeclaration(m->module, xeAbsID, Tys);
                Assert(Fn);
                llvm::Instruction *newInst = llvm::CallInst::Create(Fn, ci->getOperand(0), "");
                if (newInst != NULL) {
                    LLVMCopyMetadata(newInst, ci);
                    llvm::ReplaceInstWithInst(ci, newInst);
                    modifiedAny = true;
                    goto restart;
                }
            }
        }
// SPIR-V translator v15.0 doesn't support LLVM freeze instruction.
// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/1140
// Since it's used for optimization only, it's safe to just remove it.
#if ISPC_LLVM_VERSION == ISPC_LLVM_15_0
        else if (llvm::FreezeInst *freeze = llvm::dyn_cast<llvm::FreezeInst>(inst)) {
            llvm::Value *val = freeze->getOperand(0);
            freeze->replaceAllUsesWith(val);
            freeze->eraseFromParent();
            modifiedAny = true;
            goto restart;
        }
#endif
    }
    DEBUG_END_BB("LLVM intrinsics replacement");
    return modifiedAny;
}

bool ReplaceLLVMIntrinsics::runOnFunction(llvm::Function &F) {

    llvm::TimeTraceScope FuncScope("ReplaceLLVMIntrinsics::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= replaceUnspportedIntrinsics(BB);
    }
    return modifiedAny;
}

llvm::Pass *CreateReplaceLLVMIntrinsics() { return new ReplaceLLVMIntrinsics(); }

} // namespace ispc

#endif
