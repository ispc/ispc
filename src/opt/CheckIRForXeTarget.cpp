/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "CheckIRForXeTarget.h"

#include <regex>

#ifdef ISPC_XE_ENABLED

namespace ispc {

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
            } else {
                llvm::Function *func = ci->getCalledFunction();
                if (func == nullptr)
                    continue;
                // Check if the function name corresponds to the unsupported pattern
                std::string funcName = func->getName().str();
                static const std::regex unsupportedAtomicFuncs(
                    "__atomic_f(add|sub|min|max)_(float|double)_global|"
                    "__atomic_f(add|sub|min|max)_uniform_(float|double)_global");
                if (std::regex_match(funcName, unsupportedAtomicFuncs)) {
                    Error(pos, "This atomic operation is not supported for FP types on Xe target");
                }
            }
        }
        // Report error if double type is not supported by the target
        if (!g->target->hasFp64Support()) {
            for (int i = 0; i < (int)inst->getNumOperands(); ++i) {
                llvm::Type *t = inst->getOperand(i)->getType();
                // No need to check for double pointer types in opaque pointers mode.
                if (t->isDoubleTy()) {
                    Error(pos, "\'double\' type is not supported by the target\n");
                }
            }
        }
    }
    DEBUG_END_BB("CheckIRForXeTarget");
    return modifiedAny;
}

llvm::PreservedAnalyses CheckIRForXeTarget::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("CheckIRForXeTarget::run", F.getName());

    for (llvm::BasicBlock &BB : F) {
        checkAndFixIRForXe(BB);
    }
    return llvm::PreservedAnalyses::all();
}

} // namespace ispc

#endif
