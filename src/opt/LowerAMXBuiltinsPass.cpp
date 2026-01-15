/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause

  ISPC pass that lowers AMX builtins into LLVM X86 AMX intrinsics. This extra step is needed since LLVM AMX intrinsics
  require immediate/constant value for tile operands.
*/

#include "LowerAMXBuiltinsPass.h"
#include "builtins-decl.h"
#include "target_enums.h"

namespace ispc {

// Helper to trace through a load instruction to find a constant value.
// This handles the O0 pattern where constants are stored to allocas and then loaded.
// It also follows single-predecessor chains to find stores in predecessor blocks.
static llvm::ConstantInt *lGetConstantThroughLoad(llvm::Value *V) {
    auto *LI = llvm::dyn_cast<llvm::LoadInst>(V);
    if (!LI) {
        return nullptr;
    }

    llvm::Value *Ptr = LI->getPointerOperand()->stripPointerCasts();
    auto *AI = llvm::dyn_cast<llvm::AllocaInst>(Ptr);
    if (!AI) {
        return nullptr;
    }

    // Helper to extract constant from a store instruction.
    auto extractConstant = [](llvm::StoreInst *SI) -> llvm::ConstantInt * {
        llvm::Value *StoredVal = SI->getValueOperand();
        llvm::Type *TargetType = StoredVal->getType();

        // Trace through cast instructions (trunc, zext, sext) to find the constant.
        while (auto *Cast = llvm::dyn_cast<llvm::CastInst>(StoredVal)) {
            StoredVal = Cast->getOperand(0);
        }

        auto *CI = llvm::dyn_cast<llvm::ConstantInt>(StoredVal);
        if (!CI) {
            return nullptr;
        }

        // Return constant with the correct target type (e.g., i8 for tile IDs).
        return llvm::ConstantInt::get(llvm::cast<llvm::IntegerType>(TargetType), CI->getZExtValue());
    };

    // Search backward in a range for a store to our alloca.
    auto findStoreInRange = [AI](llvm::BasicBlock::reverse_iterator Begin,
                                 llvm::BasicBlock::reverse_iterator End) -> llvm::StoreInst * {
        for (auto It = Begin; It != End; ++It) {
            if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(&*It)) {
                if (SI->getPointerOperand()->stripPointerCasts() == AI) {
                    return SI;
                }
            }
        }
        return nullptr;
    };

    // First, search in the same basic block (starting from the load going backward).
    llvm::BasicBlock *CurrentBB = LI->getParent();
    if (auto *SI = findStoreInRange(LI->getReverseIterator(), CurrentBB->rend())) {
        return extractConstant(SI);
    }

    // Not found in same block - follow single-predecessor chain.
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> Visited;
    Visited.insert(CurrentBB);

    while (llvm::BasicBlock *PredBB = CurrentBB->getSinglePredecessor()) {
        if (!Visited.insert(PredBB).second) {
            // Already visited - loop detected.
            break;
        }

        // Search the entire predecessor block from end to beginning.
        if (auto *SI = findStoreInRange(PredBB->rbegin(), PredBB->rend())) {
            return extractConstant(SI);
        }

        CurrentBB = PredBB;
    }

    return nullptr;
}

static llvm::ConstantInt *lValidateAMXTileArgument(llvm::CallInst *CI, llvm::Value *V) {
    auto *ConstIntV = llvm::dyn_cast<llvm::ConstantInt>(V);

    // If not directly a constant, try to trace through load from alloca (O0 pattern).
    if (!ConstIntV) {
        ConstIntV = lGetConstantThroughLoad(V);
    }

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

    if (auto *CTile = lValidateAMXTileArgument(CI, TileOp)) {
        llvm::SmallVector<llvm::Value *, 1> Ops({CTile});
        return builder.CreateIntrinsic(llvm::Type::getVoidTy(builder.getContext()), llvm::Intrinsic::x86_tilezero, Ops);
    } else {
        return nullptr;
    }
}

static llvm::CallInst *lLowerAMXLoadStoreBuiltin(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    auto *CTile = lValidateAMXTileArgument(CI, CI->getArgOperand(0));
    if (!CTile) {
        return nullptr;
    }

    llvm::Intrinsic::ID LoadStoreID = 0;
    llvm::StringRef FnName = CI->getCalledFunction()->getName();
    if (FnName == builtin::__ispc_amx_tile_load) {
        LoadStoreID = llvm::Intrinsic::x86_tileloadd64;
    } else if (FnName == builtin::__ispc_amx_tile_load_t1) {
        LoadStoreID = llvm::Intrinsic::x86_tileloaddt164;
    } else if (FnName == builtin::__ispc_amx_tile_store) {
        LoadStoreID = llvm::Intrinsic::x86_tilestored64;
    } else {
        Assert(false);
    }

    llvm::SmallVector<llvm::Value *, 3> Ops({CTile, CI->getArgOperand(1), CI->getArgOperand(2)});
    return builder.CreateIntrinsic(llvm::Type::getVoidTy(builder.getContext()), LoadStoreID, Ops);
}

static llvm::CallInst *lLowerAMXDotProductBuiltin(llvm::CallInst *CI) {
    llvm::IRBuilder<> builder(CI);
    Assert(CI->arg_size() == 3);

    auto *DstTile = lValidateAMXTileArgument(CI, CI->getArgOperand(0));
    auto *Src1Tile = lValidateAMXTileArgument(CI, CI->getArgOperand(1));
    auto *Src2Tile = lValidateAMXTileArgument(CI, CI->getArgOperand(2));

    if (!DstTile || !Src1Tile || !Src2Tile) {
        return nullptr;
    }

    llvm::Intrinsic::ID DotProdID = 0;
    llvm::StringRef FnName = CI->getCalledFunction()->getName();
    if (FnName == builtin::__ispc_amx_dpbssd) {
        DotProdID = llvm::Intrinsic::x86_tdpbssd;
    } else if (FnName == builtin::__ispc_amx_dpbsud) {
        DotProdID = llvm::Intrinsic::x86_tdpbsud;
    } else if (FnName == builtin::__ispc_amx_dpbusd) {
        DotProdID = llvm::Intrinsic::x86_tdpbusd;
    } else if (FnName == builtin::__ispc_amx_dpbuud) {
        DotProdID = llvm::Intrinsic::x86_tdpbuud;
    } else if (FnName == builtin::__ispc_amx_dpbf16ps) {
        DotProdID = llvm::Intrinsic::x86_tdpbf16ps;
    } else if (FnName == builtin::__ispc_amx_dpfp16ps) {
        DotProdID = llvm::Intrinsic::x86_tdpfp16ps;
    } else {
        Assert(false);
    }

    llvm::SmallVector<llvm::Value *, 3> Ops({DstTile, Src1Tile, Src2Tile});
    return builder.CreateIntrinsic(llvm::Type::getVoidTy(builder.getContext()), DotProdID, Ops);
}

static bool lRunOnBasicBlock(llvm::BasicBlock &BB) {
    bool Modified = false;
    for (llvm::BasicBlock::iterator iter = BB.begin(), e = BB.end(); iter != e;) {
        if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(&*(iter++))) {
            llvm::Function *Callee = CI->getCalledFunction();
            if (!Callee) {
                continue;
            }

            llvm::StringRef FnName = Callee->getName();

            // Check for __ispc_amx_not_supported marker - emit error for unsupported AMX on this target
            if (FnName == builtin::__ispc_amx_not_supported) {
                SourcePos pos;
                LLVMGetSourcePosFromMetadata(CI, &pos);
                Error(pos, "Some AMX functions used in the source are not supported on the current target \"%s\".",
                      ISPCTargetToString(g->target->getISPCTarget()).c_str());
                CI->eraseFromParent();
                Modified = true;
                continue;
            }

            // Handle AMX builtins that need lowering
            if (FnName.starts_with("__ispc_amx")) {
                llvm::CallInst *D = nullptr;
                if (FnName == builtin::__ispc_amx_tile_zero) {
                    D = lLowerAMXTileZeroBuiltin(CI);
                } else if (FnName == builtin::__ispc_amx_tile_load || FnName == builtin::__ispc_amx_tile_load_t1 ||
                           FnName == builtin::__ispc_amx_tile_store) {
                    D = lLowerAMXLoadStoreBuiltin(CI);
                } else if (FnName.starts_with("__ispc_amx_dp")) {
                    D = lLowerAMXDotProductBuiltin(CI);
                }

                if (D) {
                    D->setDebugLoc(CI->getDebugLoc());
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
