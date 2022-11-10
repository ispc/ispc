/*
  Copyright (c) 2022, Intel Corporation
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

#include "CheckIRForXeTarget.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {

char CheckIRForXeTarget::ID = 0;

bool CheckIRForXeTarget::checkAndFixIRForXe(llvm::BasicBlock &bb) {
    DEBUG_START_BB("CheckIRForXeTarget");
    bool modifiedAny = false;
    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        SourcePos pos;
        LLVMGetSourcePosFromMetadata(inst, &pos);
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            if (llvm::GenXIntrinsic::getGenXIntrinsicID(ci) == llvm::GenXIntrinsic::genx_lsc_prefetch_stateless) {
                // If prefetch is supported, fix data size parameter
                Assert(ci->arg_size() > 6);
                llvm::Value *dataSizeVal = ci->getArgOperand(6);
                llvm::ConstantInt *dataSizeConst = llvm::dyn_cast<llvm::ConstantInt>(dataSizeVal);
                Assert(dataSizeConst && (dataSizeConst->getBitWidth() == 8));
                int dataSizeNum = dataSizeConst->getSExtValue();
                // 0: invalid
                // 1: d8
                // 2: d16
                // 3: d32
                // 4: d64
                // Valid user's input is 1, 2, 4, 8
                int8_t genSize = 3;
                switch (dataSizeNum) {
                case 1:
                    genSize = 1;
                    break;
                case 2:
                    genSize = 2;
                    break;
                case 4:
                    genSize = 3;
                    break;
                case 8:
                    genSize = 4;
                    break;
                default:
                    Error(pos, "Incorrect data size argument for \'prefetch\'. Valid values are 1, 2, 4, 8");
                }
                llvm::Value *dataSizeGen = llvm::ConstantInt::get(LLVMTypes::Int8Type, genSize);
                ci->setArgOperand(6, dataSizeGen);
            }
        }
        // Report error if double type is not supported by the target
        if (!g->target->hasFp64Support()) {
            for (int i = 0; i < (int)inst->getNumOperands(); ++i) {
                llvm::Type *t = inst->getOperand(i)->getType();
                if (t == LLVMTypes::DoubleType || t == LLVMTypes::DoublePointerType ||
                    t == LLVMTypes::DoubleVectorType || t == LLVMTypes::DoubleVectorPointerType) {
                    Error(pos, "\'double\' type is not supported by the target\n");
                }
            }
        }
    }
    DEBUG_END_BB("CheckIRForXeTarget");
    return modifiedAny;
}

bool CheckIRForXeTarget::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("CheckIRForXeTarget::runOnFunction", F.getName());

    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= checkAndFixIRForXe(BB);
    }
    return modifiedAny;
}

llvm::Pass *CreateCheckIRForXeTarget() { return new CheckIRForXeTarget(); }

} // namespace ispc

#endif
