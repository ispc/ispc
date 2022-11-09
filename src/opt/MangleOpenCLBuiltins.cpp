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

#include "MangleOpenCLBuiltins.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {

char MangleOpenCLBuiltins::ID = 0;

static std::string mangleMathOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    llvm::Type *retType = func.getReturnType();
    std::string funcName = func.getName().str();
    std::vector<llvm::Type *> ArgTy;
    // spirv OpenCL builtins are used for double types only
    Assert(retType->isVectorTy() && llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()->isDoubleTy() ||
           retType->isSingleValueType() && retType->isDoubleTy());
    if (retType->isVectorTy() && llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()->isDoubleTy()) {
        // Get vector width from retType. Required width may be different from target width
        // for example for 32-width targets
        ArgTy.push_back(llvm::FixedVectorType::get(LLVMTypes::DoubleType,
                                                   llvm::dyn_cast<llvm::FixedVectorType>(retType)->getNumElements()));
        // _DvWIDTH suffix is used in target file to differentiate scalar
        // and vector versions of intrinsics. Here we remove this
        // suffix and mangle the name.
        size_t pos = funcName.find("_DvWIDTH");
        if (pos != std::string::npos) {
            funcName.erase(pos, 8);
        }
    } else if (retType->isSingleValueType() && retType->isDoubleTy()) {
        ArgTy.push_back(LLVMTypes::DoubleType);
    }
    mangleOpenClBuiltin(funcName, ArgTy, mangledName);
    return mangledName;
}

static std::string manglePrintfOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName() == "__spirv_ocl_printf" && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    mangleOpenClBuiltin(func.getName().str(), func.getArg(0)->getType(), mangledName);
    return mangledName;
}

static std::string mangleOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    if (func.getName() == "__spirv_ocl_printf")
        return manglePrintfOCLBuiltin(func);
    return mangleMathOCLBuiltin(func);
}

static std::string mangleSPIRVBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_") && "wrong argument: spirv builtin is expected");
    std::string mangledName;
    mangleOpenClBuiltin(func.getName().str(), func.getArg(0)->getType(), mangledName);
    return mangledName;
}

bool MangleOpenCLBuiltins::mangleOpenCLBuiltins(llvm::BasicBlock &bb) {
    DEBUG_START_BB("MangleOpenCLBuiltins");
    bool modifiedAny = false;
    for (llvm::BasicBlock::iterator I = bb.begin(), E = --bb.end(); I != E; ++I) {
        llvm::Instruction *inst = &*I;
        if (llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst)) {
            llvm::Function *func = ci->getCalledFunction();
            if (func == NULL)
                continue;
            if (func->getName().startswith("__spirv_")) {
                std::string mangledName;
                if (func->getName().startswith("__spirv_ocl")) {
                    mangledName = mangleOCLBuiltin(*func);
                } else {
                    mangledName = mangleSPIRVBuiltin(*func);
                }
                func->setName(mangledName);
                modifiedAny = true;
            }
        }
    }
    DEBUG_END_BB("MangleOpenCLBuiltins");

    return modifiedAny;
}

bool MangleOpenCLBuiltins::runOnFunction(llvm::Function &F) {
    llvm::TimeTraceScope FuncScope("MangleOpenCLBuiltins::runOnFunction", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= mangleOpenCLBuiltins(BB);
    }
    return modifiedAny;
}

llvm::Pass *CreateMangleOpenCLBuiltins() { return new MangleOpenCLBuiltins(); }

} // namespace ispc

#endif