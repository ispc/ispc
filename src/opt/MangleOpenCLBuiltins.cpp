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

#include "MangleOpenCLBuiltins.h"

#ifdef ISPC_XE_ENABLED

namespace ispc {

char MangleOpenCLBuiltins::ID = 0;

static std::string mangleMathOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    llvm::Type *retType = func.getReturnType();
    // spirv OpenCL builtins are used for double types only
    Assert(retType->isVectorTy() && llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()->isDoubleTy() ||
           retType->isSingleValueType() && retType->isDoubleTy());

    std::string funcName = func.getName().str();
    std::vector<llvm::Type *> ArgTy;
#if ISPC_LLVM_VERSION == ISPC_LLVM_15_0
    std::vector<SPIRV::PointerIndirectPair> PointerElementTys;
#endif
    // _DvWIDTH suffix is used in target file to differentiate scalar
    // and vector versions of intrinsics. Here we remove this
    // suffix and mangle the name.
    size_t pos = funcName.find("_DvWIDTH");
    bool isVaryingFunc = pos != std::string::npos;
    if (isVaryingFunc) {
        funcName.erase(pos, 8);
    }
    for (auto &arg : func.args()) {
        ArgTy.push_back(arg.getType());
        // In LLVM15 SPIR-V translator requires to pass pointer type information to mangleBuiltin
        // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/commit/0eb9a7d2937542e1f95a4e1f9aa9850e669dc45f
        // It changes again in LLVM16 SPIR-V translator where TypedPointerType is required
        // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/commit/42cf770344bb8d0a32db1ec892bee63f43d793b1
#if ISPC_LLVM_VERSION == ISPC_LLVM_15_0
        SPIRV::PointerIndirectPair PtrElemTy;
        if (arg.getType()->isPointerTy()) {
            PtrElemTy.setPointer(isVaryingFunc ? LLVMTypes::DoubleVectorType : LLVMTypes::DoubleType);
        }
        PointerElementTys.push_back(PtrElemTy);
#endif
    }

    mangleOpenClBuiltin(funcName, ArgTy,
#if ISPC_LLVM_VERSION == ISPC_LLVM_15_0
                        PointerElementTys,
#endif
                        mangledName);
    return mangledName;
}

static std::string manglePrintfOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName() == "__spirv_ocl_printf" && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    mangleOpenClBuiltin(func.getName().str(), func.getArg(0)->getType(),
#if ISPC_LLVM_VERSION == ISPC_LLVM_15_0
                        // For spirv_ocl_printf builtin the argument is always i8*
                        SPIRV::PointerIndirectPair(LLVMTypes::Int8Type),
#endif
                        mangledName);
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
    mangleOpenClBuiltin(func.getName().str(), func.getArg(0)->getType(),
#if ISPC_LLVM_VERSION == ISPC_LLVM_15_0
                        // spirv builtins doesn't have pointer arguments
                        {},
#endif
                        mangledName);
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