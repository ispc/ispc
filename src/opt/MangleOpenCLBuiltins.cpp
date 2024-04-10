/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "MangleOpenCLBuiltins.h"

#ifdef ISPC_XE_ENABLED
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
#include "llvm/IR/TypedPointerType.h"
#endif

namespace ispc {

static std::string mangleMathOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    llvm::Type *retType = func.getReturnType();
    // spirv OpenCL builtins are used for half/float/double types only
    llvm::Type *retElementType = (retType->isVectorTy() && llvm::isa<llvm::FixedVectorType>(retType))
                                     ? llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()
                                     : retType;
    Assert(retElementType->isHalfTy() || retElementType->isFloatTy() || retElementType->isDoubleTy());

    std::string funcName = func.getName().str();
    std::vector<llvm::Type *> ArgTy;
#if ISPC_LLVM_VERSION == ISPC_LLVM_15_0
    std::vector<SPIRV::PointerIndirectPair> PointerElementTys;
#endif
    // _DvWIDTH suffix is used in target file to differentiate scalar (DvWIDTH1<type>)
    // and vector (DvWIDTH<type>) versions of intrinsics for different types. Here we remove this
    // suffix and mangle the name.
    // We don't use mangled names in target file for 2 reasons:
    // 1. target file is used for different vector widths
    // 2. spirv builtins may be used as part of macros (e.g. see xe_double_math)
    size_t pos = funcName.find("_DvWIDTH");
    bool suffixPos = pos != std::string::npos;
    if (suffixPos) {
        funcName.erase(pos, funcName.length() - pos);
    }
    for (auto &arg : func.args()) {
        // In LLVM15 SPIR-V translator requires to pass pointer type information to mangleBuiltin
        // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/commit/0eb9a7d2937542e1f95a4e1f9aa9850e669dc45f
        // It changes again in LLVM16 SPIR-V translator where TypedPointerType is required
        // https://github.com/KhronosGroup/SPIRV-LLVM-Translator/commit/42cf770344bb8d0a32db1ec892bee63f43d793b1
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
        if (arg.getType()->isPointerTy()) {
            // In SPIR-V OpenCL builtins it's safe to assume that pointer argument is either pointer to <type>
            // or pointer to <WIDTH x type> which is basically type of retType.
            ArgTy.push_back(llvm::TypedPointerType::get(retType, 0));
        } else {
            ArgTy.push_back(arg.getType());
        }
#elif ISPC_LLVM_VERSION == ISPC_LLVM_15_0
        ArgTy.push_back(arg.getType());
        SPIRV::PointerIndirectPair PtrElemTy;
        if (arg.getType()->isPointerTy()) {
            PtrElemTy.setPointer(retType);
        }
        PointerElementTys.push_back(PtrElemTy);
#else
        ArgTy.push_back(arg.getType());
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

std::string mangleSPIRVBuiltin(const llvm::Function &func) {
    Assert(func.getName().startswith("__spirv_") && "wrong argument: spirv builtin is expected");
    std::string mangledName;
    std::vector<llvm::Type *> tyArgs;
    for (const auto &arg : func.args()) {
        tyArgs.push_back(arg.getType());
    }
    mangleOpenClBuiltin(func.getName().str(), tyArgs,
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
            if (func == nullptr)
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

llvm::PreservedAnalyses MangleOpenCLBuiltins::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("MangleOpenCLBuiltins::run", F.getName());
    for (llvm::BasicBlock &BB : F) {
        mangleOpenCLBuiltins(BB);
    }
    return llvm::PreservedAnalyses::all();
}

} // namespace ispc

#endif
