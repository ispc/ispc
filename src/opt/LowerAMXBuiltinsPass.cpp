/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause

  ISPC pass that lowers AMX builtins into LLVM X86 AMX intrinsics. This extra step is needed since LLVM AMX intrinsics
  require immediate/constant value for tile operands.
*/

#include "LowerAMXBuiltinsPass.h"

namespace ispc {

static llvm::ConstantInt *validateAMXTileArgument(llvm::CallInst *CI, llvm::Value *V) {
    auto *ConstIntV = llvm::dyn_cast<llvm::ConstantInt>(V);

    if (!ConstIntV) {
        SourcePos pos;
        LLVMGetSourcePosFromMetadata(CI, &pos);
        Error(pos, "AMX tile argument must be compile time constant.");
        return nullptr;
    }

    uint64_t TileID = ConstIntV->getZExtValue();

    if (TileID > 7) {
        SourcePos pos;
        LLVMGetSourcePosFromMetadata(CI, &pos);
        Error(pos, "AMX tile argument value must be between 0-7.");
        return nullptr;
    }

    return ConstIntV;
}

static llvm::CallInst *lLowerAMXTileZeroBuiltin(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 1);

    llvm::Value *TileOp = CI->getArgOperand(0);

    if (auto *CTile = validateAMXTileArgument(CI, TileOp)) {
        llvm::SmallVector<llvm::Value *, 1> Ops({CTile});
        return builder.CreateIntrinsic(llvm::Type::getVoidTy(builder.getContext()), llvm::Intrinsic::x86_tilezero, Ops);
    } else {
        return nullptr;
    }
}

static llvm::CallInst *lLowerAMXLoadStoreBuiltin(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    auto *CTile = validateAMXTileArgument(CI, CI->getArgOperand(0));
    if (!CTile)
        return nullptr;

    llvm::Intrinsic::ID LoadStoreID = 0;
    llvm::StringRef FnName = CI->getCalledFunction()->getName();
    if (FnName.equals_insensitive("__ispc_amx_load"))
        LoadStoreID = llvm::Intrinsic::x86_tileloadd64;
    else if (FnName.equals_insensitive("__ispc_amx_load_t1"))
        LoadStoreID = llvm::Intrinsic::x86_tileloaddt164;
    else if (FnName.equals_insensitive("__ispc_amx_store"))
        LoadStoreID = llvm::Intrinsic::x86_tilestored64;
    else
        Assert(false);

    llvm::SmallVector<llvm::Value *, 3> Ops({CTile, CI->getArgOperand(1), CI->getArgOperand(2)});
    return builder.CreateIntrinsic(llvm::Type::getVoidTy(builder.getContext()), LoadStoreID, Ops);
}

static llvm::CallInst *lLowerAMXDotProductBuiltin(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    auto *DstTile = validateAMXTileArgument(CI, CI->getArgOperand(0));
    auto *Src1Tile = validateAMXTileArgument(CI, CI->getArgOperand(1));
    auto *Src2Tile = validateAMXTileArgument(CI, CI->getArgOperand(2));

    if (!DstTile || !Src1Tile || !Src2Tile)
        return nullptr;

    llvm::Intrinsic::ID DotProdID = 0;
    llvm::StringRef FnName = CI->getCalledFunction()->getName();
    if (FnName.equals_insensitive("__ispc_amx_dpbssd"))
        DotProdID = llvm::Intrinsic::x86_tdpbssd;
    else if (FnName.equals_insensitive("__ispc_amx_dpbsud"))
        DotProdID = llvm::Intrinsic::x86_tdpbsud;
    else if (FnName.equals_insensitive("__ispc_amx_dpbusd"))
        DotProdID = llvm::Intrinsic::x86_tdpbusd;
    else if (FnName.equals_insensitive("__ispc_amx_dpbuud"))
        DotProdID = llvm::Intrinsic::x86_tdpbuud;
    else if (FnName.equals_insensitive("__ispc_amx_dpbf16ps"))
        DotProdID = llvm::Intrinsic::x86_tdpbf16ps;
    else if (FnName.equals_insensitive("__ispc_amx_dpfp16ps"))
        DotProdID = llvm::Intrinsic::x86_tdpfp16ps;
    else
        Assert(false);

    llvm::SmallVector<llvm::Value *, 3> Ops({DstTile, Src1Tile, Src2Tile});
    return builder.CreateIntrinsic(llvm::Type::getVoidTy(builder.getContext()), DotProdID, Ops);
}

static bool lRunOnBasicBlock(llvm::BasicBlock &BB) {
    bool Modified = false;
    for (llvm::BasicBlock::iterator iter = BB.begin(), e = BB.end(); iter != e;) {
        if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(&*(iter++))) {
            llvm::Function *Callee = CI->getCalledFunction();
            if (Callee && Callee->getName().starts_with("__ispc_amx")) {
                llvm::CallInst *D = nullptr;
                if (Callee->getName().starts_with("__ispc_amx_zero")) {
                    D = lLowerAMXTileZeroBuiltin(CI);
                } else if (Callee->getName().starts_with("__ispc_amx_load")) {
                    D = lLowerAMXLoadStoreBuiltin(CI);
                } else if (Callee->getName().starts_with("__ispc_amx_store")) {
                    D = lLowerAMXLoadStoreBuiltin(CI);
                } else if (Callee->getName().starts_with("__ispc_amx_dp")) {
                    D = lLowerAMXDotProductBuiltin(CI);
                }

                if (D) {
                    CI->replaceAllUsesWith(D);
                    CI->eraseFromParent();
                    Modified = true;
                }
            }
        }
    }
    return Modified;
}

llvm::PreservedAnalyses LowerAMXBuiltinsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("LowerAMXBuiltinsPass::run", F.getName());
    bool modified = false;
    for (llvm::BasicBlock &BB : F) {
        modified |= lRunOnBasicBlock(BB);
    }
    if (modified) {
        llvm::PreservedAnalyses PA;
        PA.preserveSet<llvm::CFGAnalyses>();
        return PA;
    } else {
        return llvm::PreservedAnalyses::all();
    }
}
} // namespace ispc
