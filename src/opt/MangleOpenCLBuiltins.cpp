/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "MangleOpenCLBuiltins.h"

#ifdef ISPC_XE_ENABLED
#include "llvm/IR/TypedPointerType.h"

namespace ispc {

static std::string mangleMathOCLBuiltin(const llvm::Function &func) {
    Assert(func.getName().starts_with("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    std::string mangledName;
    llvm::Type *retType = func.getReturnType();
    // spirv OpenCL builtins are used for half/float/double types only
    llvm::Type *retElementType = (retType->isVectorTy() && llvm::isa<llvm::FixedVectorType>(retType))
                                     ? llvm::dyn_cast<llvm::FixedVectorType>(retType)->getElementType()
                                     : retType;
    Assert(retElementType->isHalfTy() || retElementType->isFloatTy() || retElementType->isDoubleTy());

    std::string funcName = func.getName().str();
    std::vector<llvm::Type *> ArgTy;
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
        if (arg.getType()->isPointerTy()) {
            // In SPIR-V OpenCL builtins it's safe to assume that pointer argument is either pointer to <type>
            // or pointer to <WIDTH x type> which is basically type of retType.
            ArgTy.push_back(llvm::TypedPointerType::get(retType, 0));
        } else {
            ArgTy.push_back(arg.getType());
        }
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
    Assert(func.getName().starts_with("__spirv_ocl") && "wrong argument: ocl builtin is expected");
    if (func.getName() == "__spirv_ocl_printf")
        return manglePrintfOCLBuiltin(func);
    return mangleMathOCLBuiltin(func);
}

std::string mangleSPIRVBuiltin(const llvm::Function &func) {
    Assert(func.getName().starts_with("__spirv_") && "wrong argument: spirv builtin is expected");
    std::string mangledName;
    std::vector<llvm::Type *> tyArgs;
    for (const auto &arg : func.args()) {
        tyArgs.push_back(arg.getType());
    }
    mangleOpenClBuiltin(func.getName().str(), tyArgs, mangledName);
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
            if (func->getName().starts_with("__spirv_")) {
                std::string mangledName;
                if (func->getName().starts_with("__spirv_ocl")) {
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
