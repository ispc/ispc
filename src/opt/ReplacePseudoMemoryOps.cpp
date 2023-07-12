/*
  Copyright (c) 2022-2023, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ReplacePseudoMemoryOps.h"

namespace ispc {

/** This routine attempts to determine if the given pointer in lvalue is
    pointing to stack-allocated memory.  It's conservative in that it
    should never return true for non-stack allocated memory, but may return
    false for memory that actually is stack allocated.  The basic strategy
    is to traverse through the operands and see if the pointer originally
    comes from an AllocaInst.
*/
static bool lIsSafeToBlend(llvm::Value *lvalue) {
    llvm::BitCastInst *bc = llvm::dyn_cast<llvm::BitCastInst>(lvalue);
    if (bc != nullptr)
        return lIsSafeToBlend(bc->getOperand(0));
    else {
        llvm::AllocaInst *ai = llvm::dyn_cast<llvm::AllocaInst>(lvalue);
        if (ai) {
            llvm::Type *type = ai->getAllocatedType();
            llvm::ArrayType *at;
            while ((at = llvm::dyn_cast<llvm::ArrayType>(type))) {
                type = at->getElementType();
            }
            llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(type);
            return (vt != nullptr && (int)vt->getNumElements() == g->target->getVectorWidth());
        } else {
            llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(lvalue);
            if (gep != nullptr)
                return lIsSafeToBlend(gep->getOperand(0));
            else
                return false;
        }
    }
}

static bool lReplacePseudoMaskedStore(llvm::CallInst *callInst) {
    struct LMSInfo {
        LMSInfo(const char *pname, const char *bname, const char *msname) {
            pseudoFunc = m->module->getFunction(pname);
            blendFunc = m->module->getFunction(bname);
            maskedStoreFunc = m->module->getFunction(msname);
            Assert(pseudoFunc != nullptr && blendFunc != nullptr && maskedStoreFunc != nullptr);
        }
        llvm::Function *pseudoFunc;
        llvm::Function *blendFunc;
        llvm::Function *maskedStoreFunc;
    };

    LMSInfo msInfo[] = {
        LMSInfo("__pseudo_masked_store_i8", "__masked_store_blend_i8", "__masked_store_i8"),
        LMSInfo("__pseudo_masked_store_i16", "__masked_store_blend_i16", "__masked_store_i16"),
        LMSInfo("__pseudo_masked_store_half", "__masked_store_blend_half", "__masked_store_half"),
        LMSInfo("__pseudo_masked_store_i32", "__masked_store_blend_i32", "__masked_store_i32"),
        LMSInfo("__pseudo_masked_store_float", "__masked_store_blend_float", "__masked_store_float"),
        LMSInfo("__pseudo_masked_store_i64", "__masked_store_blend_i64", "__masked_store_i64"),
        LMSInfo("__pseudo_masked_store_double", "__masked_store_blend_double", "__masked_store_double")};
    LMSInfo *info = nullptr;
    for (unsigned int i = 0; i < sizeof(msInfo) / sizeof(msInfo[0]); ++i) {
        if (msInfo[i].pseudoFunc != nullptr && callInst->getCalledFunction() == msInfo[i].pseudoFunc) {
            info = &msInfo[i];
            break;
        }
    }
    if (info == nullptr)
        return false;

    llvm::Value *lvalue = callInst->getArgOperand(0);
    llvm::Value *rvalue = callInst->getArgOperand(1);
    llvm::Value *mask = callInst->getArgOperand(2);

    // We need to choose between doing the load + blend + store trick,
    // or serializing the masked store.  Even on targets with a native
    // masked store instruction, this is preferable since it lets us
    // keep values in registers rather than going out to the stack.
    bool doBlend = (!g->opt.disableBlendedMaskedStores && lIsSafeToBlend(lvalue));

    // Generate the call to the appropriate masked store function and
    // replace the __pseudo_* one with it.
    llvm::Function *fms = doBlend ? info->blendFunc : info->maskedStoreFunc;
    llvm::Instruction *inst = LLVMCallInst(fms, lvalue, rvalue, mask, "", callInst);
    LLVMCopyMetadata(inst, callInst);

    callInst->eraseFromParent();
    return true;
}

static bool lReplacePseudoGS(llvm::CallInst *callInst) {
    struct LowerGSInfo {
        LowerGSInfo(const char *pName, const char *aName, bool ig, bool ip) : isGather(ig), isPrefetch(ip) {
            pseudoFunc = m->module->getFunction(pName);
            actualFunc = m->module->getFunction(aName);
        }
        llvm::Function *pseudoFunc;
        llvm::Function *actualFunc;
        const bool isGather;
        const bool isPrefetch;
    };

    LowerGSInfo lgsInfo[] = {
        LowerGSInfo("__pseudo_gather32_i8",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather32_generic_i8" : "__gather32_i8", true,
                    false),
        LowerGSInfo("__pseudo_gather32_i16",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather32_generic_i16" : "__gather32_i16", true,
                    false),
        LowerGSInfo("__pseudo_gather32_half",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather32_generic_half" : "__gather32_half",
                    true, false),
        LowerGSInfo("__pseudo_gather32_i32",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather32_generic_i32" : "__gather32_i32", true,
                    false),
        LowerGSInfo("__pseudo_gather32_float",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather32_generic_float" : "__gather32_float",
                    true, false),
        LowerGSInfo("__pseudo_gather32_i64",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather32_generic_i64" : "__gather32_i64", true,
                    false),
        LowerGSInfo("__pseudo_gather32_double",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather32_generic_double" : "__gather32_double",
                    true, false),

        LowerGSInfo("__pseudo_gather64_i8",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather64_generic_i8" : "__gather64_i8", true,
                    false),
        LowerGSInfo("__pseudo_gather64_i16",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather64_generic_i16" : "__gather64_i16", true,
                    false),
        LowerGSInfo("__pseudo_gather64_half",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather64_generic_half" : "__gather64_half",
                    true, false),
        LowerGSInfo("__pseudo_gather64_i32",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather64_generic_i32" : "__gather64_i32", true,
                    false),
        LowerGSInfo("__pseudo_gather64_float",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather64_generic_float" : "__gather64_float",
                    true, false),
        LowerGSInfo("__pseudo_gather64_i64",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather64_generic_i64" : "__gather64_i64", true,
                    false),
        LowerGSInfo("__pseudo_gather64_double",
                    g->target->hasGather() && g->opt.disableGathers ? "__gather64_generic_double" : "__gather64_double",
                    true, false),

        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i8", "__gather_factored_base_offsets32_i8", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i16", "__gather_factored_base_offsets32_i16", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_half", "__gather_factored_base_offsets32_half", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i32", "__gather_factored_base_offsets32_i32", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_float", "__gather_factored_base_offsets32_float", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_i64", "__gather_factored_base_offsets32_i64", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets32_double", "__gather_factored_base_offsets32_double", true,
                    false),

        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i8", "__gather_factored_base_offsets64_i8", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i16", "__gather_factored_base_offsets64_i16", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_half", "__gather_factored_base_offsets64_half", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i32", "__gather_factored_base_offsets64_i32", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_float", "__gather_factored_base_offsets64_float", true,
                    false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_i64", "__gather_factored_base_offsets64_i64", true, false),
        LowerGSInfo("__pseudo_gather_factored_base_offsets64_double", "__gather_factored_base_offsets64_double", true,
                    false),

        LowerGSInfo("__pseudo_gather_base_offsets32_i8", "__gather_base_offsets32_i8", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_i16", "__gather_base_offsets32_i16", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_half", "__gather_base_offsets32_half", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_i32", "__gather_base_offsets32_i32", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_float", "__gather_base_offsets32_float", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_i64", "__gather_base_offsets32_i64", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets32_double", "__gather_base_offsets32_double", true, false),

        LowerGSInfo("__pseudo_gather_base_offsets64_i8", "__gather_base_offsets64_i8", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_i16", "__gather_base_offsets64_i16", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_half", "__gather_base_offsets64_half", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_i32", "__gather_base_offsets64_i32", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_float", "__gather_base_offsets64_float", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_i64", "__gather_base_offsets64_i64", true, false),
        LowerGSInfo("__pseudo_gather_base_offsets64_double", "__gather_base_offsets64_double", true, false),

        LowerGSInfo("__pseudo_scatter32_i8",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter32_generic_i8" : "__scatter32_i8",
                    false, false),
        LowerGSInfo("__pseudo_scatter32_i16",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter32_generic_i16" : "__scatter32_i16",
                    false, false),
        LowerGSInfo("__pseudo_scatter32_half",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter32_generic_half" : "__scatter32_half",
                    false, false),
        LowerGSInfo("__pseudo_scatter32_i32",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter32_generic_i32" : "__scatter32_i32",
                    false, false),
        LowerGSInfo("__pseudo_scatter32_float",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter32_generic_float"
                                                                      : "__scatter32_float",
                    false, false),
        LowerGSInfo("__pseudo_scatter32_i64",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter32_generic_i64" : "__scatter32_i64",
                    false, false),
        LowerGSInfo("__pseudo_scatter32_double",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter32_generic_double"
                                                                      : "__scatter32_double",
                    false, false),

        LowerGSInfo("__pseudo_scatter64_i8",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter64_generic_i8" : "__scatter64_i8",
                    false, false),
        LowerGSInfo("__pseudo_scatter64_i16",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter64_generic_i16" : "__scatter64_i16",
                    false, false),
        LowerGSInfo("__pseudo_scatter64_half",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter64_generic_half" : "__scatter64_half",
                    false, false),
        LowerGSInfo("__pseudo_scatter64_i32",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter64_generic_i32" : "__scatter64_i32",
                    false, false),
        LowerGSInfo("__pseudo_scatter64_float",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter64_generic_float"
                                                                      : "__scatter64_float",
                    false, false),
        LowerGSInfo("__pseudo_scatter64_i64",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter64_generic_i64" : "__scatter64_i64",
                    false, false),
        LowerGSInfo("__pseudo_scatter64_double",
                    g->target->hasScatter() && g->opt.disableScatters ? "__scatter64_generic_double"
                                                                      : "__scatter64_double",
                    false, false),

        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i8", "__scatter_factored_base_offsets32_i8", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i16", "__scatter_factored_base_offsets32_i16", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_half", "__scatter_factored_base_offsets32_half", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i32", "__scatter_factored_base_offsets32_i32", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_float", "__scatter_factored_base_offsets32_float", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_i64", "__scatter_factored_base_offsets32_i64", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets32_double", "__scatter_factored_base_offsets32_double",
                    false, false),

        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i8", "__scatter_factored_base_offsets64_i8", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i16", "__scatter_factored_base_offsets64_i16", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_half", "__scatter_factored_base_offsets64_half", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i32", "__scatter_factored_base_offsets64_i32", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_float", "__scatter_factored_base_offsets64_float", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_i64", "__scatter_factored_base_offsets64_i64", false,
                    false),
        LowerGSInfo("__pseudo_scatter_factored_base_offsets64_double", "__scatter_factored_base_offsets64_double",
                    false, false),

        LowerGSInfo("__pseudo_scatter_base_offsets32_i8", "__scatter_base_offsets32_i8", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_i16", "__scatter_base_offsets32_i16", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_half", "__scatter_base_offsets32_half", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_i32", "__scatter_base_offsets32_i32", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_float", "__scatter_base_offsets32_float", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_i64", "__scatter_base_offsets32_i64", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets32_double", "__scatter_base_offsets32_double", false, false),

        LowerGSInfo("__pseudo_scatter_base_offsets64_i8", "__scatter_base_offsets64_i8", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_i16", "__scatter_base_offsets64_i16", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_half", "__scatter_base_offsets64_half", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_i32", "__scatter_base_offsets64_i32", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_float", "__scatter_base_offsets64_float", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_i64", "__scatter_base_offsets64_i64", false, false),
        LowerGSInfo("__pseudo_scatter_base_offsets64_double", "__scatter_base_offsets64_double", false, false),

        LowerGSInfo("__pseudo_prefetch_read_varying_1", "__prefetch_read_varying_1", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_1_native", "__prefetch_read_varying_1_native", false, true),

        LowerGSInfo("__pseudo_prefetch_read_varying_2", "__prefetch_read_varying_2", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_2_native", "__prefetch_read_varying_2_native", false, true),

        LowerGSInfo("__pseudo_prefetch_read_varying_3", "__prefetch_read_varying_3", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_3_native", "__prefetch_read_varying_3_native", false, true),

        LowerGSInfo("__pseudo_prefetch_read_varying_nt", "__prefetch_read_varying_nt", false, true),
        LowerGSInfo("__pseudo_prefetch_read_varying_nt_native", "__prefetch_read_varying_nt_native", false, true),

        LowerGSInfo("__pseudo_prefetch_write_varying_1", "__prefetch_write_varying_1", false, true),
        LowerGSInfo("__pseudo_prefetch_write_varying_1_native", "__prefetch_write_varying_1_native", false, true),

        LowerGSInfo("__pseudo_prefetch_write_varying_2", "__prefetch_write_varying_2", false, true),
        LowerGSInfo("__pseudo_prefetch_write_varying_2_native", "__prefetch_write_varying_2_native", false, true),

        LowerGSInfo("__pseudo_prefetch_write_varying_3", "__prefetch_write_varying_3", false, true),
        LowerGSInfo("__pseudo_prefetch_write_varying_3_native", "__prefetch_write_varying_3_native", false, true),
    };

    llvm::Function *calledFunc = callInst->getCalledFunction();

    LowerGSInfo *info = nullptr;
    for (unsigned int i = 0; i < sizeof(lgsInfo) / sizeof(lgsInfo[0]); ++i) {
        if (lgsInfo[i].pseudoFunc != nullptr && calledFunc == lgsInfo[i].pseudoFunc) {
            info = &lgsInfo[i];
            break;
        }
    }
    if (info == nullptr)
        return false;

    Assert(info->actualFunc != nullptr);

    // Get the source position from the metadata attached to the call
    // instruction so that we can issue PerformanceWarning()s below.
    SourcePos pos;
    bool gotPosition = LLVMGetSourcePosFromMetadata(callInst, &pos);

    callInst->setCalledFunction(info->actualFunc);
    // Check for alloca and if not alloca - generate __gather and change arguments
    if (gotPosition && (g->target->getVectorWidth() > 1) && (g->opt.level > 0)) {
        if (info->isGather)
            PerformanceWarning(pos, "Gather required to load value.");
        else if (!info->isPrefetch)
            PerformanceWarning(pos, "Scatter required to store value.");
    }
    return true;
}

bool ReplacePseudoMemoryOpsPass::replacePseudoMemoryOps(llvm::BasicBlock &bb) {
    DEBUG_START_BB("ReplacePseudoMemoryOpsPass");

    bool modifiedAny = false;

    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*(iter++));
        if (callInst == nullptr || callInst->getCalledFunction() == nullptr)
            continue;

        if (lReplacePseudoGS(callInst)) {
            modifiedAny = true;
        } else if (lReplacePseudoMaskedStore(callInst)) {
            modifiedAny = true;
        }
    }

    DEBUG_END_BB("ReplacePseudoMemoryOpsPass");

    return modifiedAny;
}

llvm::PreservedAnalyses ReplacePseudoMemoryOpsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("ReplacePseudoMemoryOpsPass::run", F.getName());
    bool modifiedAny = false;
    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= replacePseudoMemoryOps(BB);
    }
    if (!modifiedAny) {
        // No changes, all analyses are preserved.
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
