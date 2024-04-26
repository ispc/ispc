/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ReplacePseudoMemoryOps.h"
#include "builtins-decl.h"

namespace ispc {

using namespace builtin;

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

struct LMSInfo {
    LMSInfo(const char *bname, const char *msname) : blend(bname), store(msname) {}
    llvm::Function *blendFunc(llvm::Module *M) const { return M->getFunction(blend); }
    llvm::Function *maskedStoreFunc(llvm::Module *M) const { return M->getFunction(store); }

  private:
    const char *blend;
    const char *store;
};

static bool lReplacePseudoMaskedStore(llvm::CallInst *callInst) {
    static std::unordered_map<std::string, LMSInfo> replacementRules = {
        {__pseudo_masked_store_i8, LMSInfo(__masked_store_blend_i8, __masked_store_i8)},
        {__pseudo_masked_store_i16, LMSInfo(__masked_store_blend_i16, __masked_store_i16)},
        {__pseudo_masked_store_half, LMSInfo(__masked_store_blend_half, __masked_store_half)},
        {__pseudo_masked_store_i32, LMSInfo(__masked_store_blend_i32, __masked_store_i32)},
        {__pseudo_masked_store_float, LMSInfo(__masked_store_blend_float, __masked_store_float)},
        {__pseudo_masked_store_i64, LMSInfo(__masked_store_blend_i64, __masked_store_i64)},
        {__pseudo_masked_store_double, LMSInfo(__masked_store_blend_double, __masked_store_double)},
    };

    auto name = callInst->getCalledFunction()->getName().str();
    auto it = replacementRules.find(name);
    if (it == replacementRules.end()) {
        // it is not a call of __pseudo function stored in replacementRules
        return false;
    }
    LMSInfo *info = &it->second;

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
    llvm::Module *M = callInst->getModule();
    llvm::Function *fms = doBlend ? info->blendFunc(M) : info->maskedStoreFunc(M);
    llvm::Instruction *inst = LLVMCallInst(fms, lvalue, rvalue, mask, "", callInst);
    LLVMCopyMetadata(inst, callInst);

    callInst->eraseFromParent();
    return true;
}

struct LowerGSInfo {
    enum class Type {
        Gather,
        Scatter,
        Prefetch,
    };
    LowerGSInfo(const char *aName, Type type) : generic(aName), name(aName), type(type) {}
    LowerGSInfo(const char *gName, const char *aName, Type type) : generic(gName), name(aName), type(type) {}
    static LowerGSInfo Gather(const char *aName) { return LowerGSInfo(aName, Type::Gather); }
    static LowerGSInfo Gather(const char *gName, const char *aName) { return LowerGSInfo(gName, aName, Type::Gather); }
    static LowerGSInfo Scatter(const char *aName) { return LowerGSInfo(aName, Type::Scatter); }
    static LowerGSInfo Scatter(const char *gName, const char *aName) {
        return LowerGSInfo(gName, aName, Type::Scatter);
    }
    static LowerGSInfo Prefetch(const char *aName) { return LowerGSInfo(aName, Type::Prefetch); }
    llvm::Function *actualFunc(llvm::Module *M) const {
        const char *func = name;
        if (isGather()) {
            func = g->target->hasGather() && g->opt.disableGathers ? generic : name;
        }
        if (isScatter()) {
            func = g->target->hasScatter() && g->opt.disableScatters ? generic : name;
        }
        return M->getFunction(func);
    }
    bool isGather() const { return type == Type::Gather; }
    bool isScatter() const { return type == Type::Scatter; }
    bool isPrefetch() const { return type == Type::Prefetch; }

  private:
    const char *generic;
    const char *name;
    Type type;
};

static bool lReplacePseudoGS(llvm::CallInst *callInst) {

    static std::unordered_map<std::string, LowerGSInfo> replacementRules = {
        {__pseudo_gather32_i8, LowerGSInfo::Gather(__gather32_generic_i8, __gather32_i8)},
        {__pseudo_gather32_i16, LowerGSInfo::Gather(__gather32_generic_i16, __gather32_i16)},
        {__pseudo_gather32_half, LowerGSInfo::Gather(__gather32_generic_half, __gather32_half)},
        {__pseudo_gather32_i32, LowerGSInfo::Gather(__gather32_generic_i32, __gather32_i32)},
        {__pseudo_gather32_float, LowerGSInfo::Gather(__gather32_generic_float, __gather32_float)},
        {__pseudo_gather32_i64, LowerGSInfo::Gather(__gather32_generic_i64, __gather32_i64)},
        {__pseudo_gather32_double, LowerGSInfo::Gather(__gather32_generic_double, __gather32_double)},

        {__pseudo_gather64_i8, LowerGSInfo::Gather(__gather64_generic_i8, __gather64_i8)},
        {__pseudo_gather64_i16, LowerGSInfo::Gather(__gather64_generic_i16, __gather64_i16)},
        {__pseudo_gather64_half, LowerGSInfo::Gather(__gather64_generic_half, __gather64_half)},
        {__pseudo_gather64_i32, LowerGSInfo::Gather(__gather64_generic_i32, __gather64_i32)},
        {__pseudo_gather64_float, LowerGSInfo::Gather(__gather64_generic_float, __gather64_float)},
        {__pseudo_gather64_i64, LowerGSInfo::Gather(__gather64_generic_i64, __gather64_i64)},
        {__pseudo_gather64_double, LowerGSInfo::Gather(__gather64_generic_double, __gather64_double)},

        {__pseudo_gather_factored_base_offsets32_i8, LowerGSInfo::Gather(__gather_factored_base_offsets32_i8)},
        {__pseudo_gather_factored_base_offsets32_i16, LowerGSInfo::Gather(__gather_factored_base_offsets32_i16)},
        {__pseudo_gather_factored_base_offsets32_half, LowerGSInfo::Gather(__gather_factored_base_offsets32_half)},
        {__pseudo_gather_factored_base_offsets32_i32, LowerGSInfo::Gather(__gather_factored_base_offsets32_i32)},
        {__pseudo_gather_factored_base_offsets32_float, LowerGSInfo::Gather(__gather_factored_base_offsets32_float)},
        {__pseudo_gather_factored_base_offsets32_i64, LowerGSInfo::Gather(__gather_factored_base_offsets32_i64)},
        {__pseudo_gather_factored_base_offsets32_double, LowerGSInfo::Gather(__gather_factored_base_offsets32_double)},

        {__pseudo_gather_factored_base_offsets64_i8, LowerGSInfo::Gather(__gather_factored_base_offsets64_i8)},
        {__pseudo_gather_factored_base_offsets64_i16, LowerGSInfo::Gather(__gather_factored_base_offsets64_i16)},
        {__pseudo_gather_factored_base_offsets64_half, LowerGSInfo::Gather(__gather_factored_base_offsets64_half)},
        {__pseudo_gather_factored_base_offsets64_i32, LowerGSInfo::Gather(__gather_factored_base_offsets64_i32)},
        {__pseudo_gather_factored_base_offsets64_float, LowerGSInfo::Gather(__gather_factored_base_offsets64_float)},
        {__pseudo_gather_factored_base_offsets64_i64, LowerGSInfo::Gather(__gather_factored_base_offsets64_i64)},
        {__pseudo_gather_factored_base_offsets64_double, LowerGSInfo::Gather(__gather_factored_base_offsets64_double)},

        {__pseudo_gather_base_offsets32_i8, LowerGSInfo::Gather(__gather_base_offsets32_i8)},
        {__pseudo_gather_base_offsets32_i16, LowerGSInfo::Gather(__gather_base_offsets32_i16)},
        {__pseudo_gather_base_offsets32_half, LowerGSInfo::Gather(__gather_base_offsets32_half)},
        {__pseudo_gather_base_offsets32_i32, LowerGSInfo::Gather(__gather_base_offsets32_i32)},
        {__pseudo_gather_base_offsets32_float, LowerGSInfo::Gather(__gather_base_offsets32_float)},
        {__pseudo_gather_base_offsets32_i64, LowerGSInfo::Gather(__gather_base_offsets32_i64)},
        {__pseudo_gather_base_offsets32_double, LowerGSInfo::Gather(__gather_base_offsets32_double)},

        {__pseudo_gather_base_offsets64_i8, LowerGSInfo::Gather(__gather_base_offsets64_i8)},
        {__pseudo_gather_base_offsets64_i16, LowerGSInfo::Gather(__gather_base_offsets64_i16)},
        {__pseudo_gather_base_offsets64_half, LowerGSInfo::Gather(__gather_base_offsets64_half)},
        {__pseudo_gather_base_offsets64_i32, LowerGSInfo::Gather(__gather_base_offsets64_i32)},
        {__pseudo_gather_base_offsets64_float, LowerGSInfo::Gather(__gather_base_offsets64_float)},
        {__pseudo_gather_base_offsets64_i64, LowerGSInfo::Gather(__gather_base_offsets64_i64)},
        {__pseudo_gather_base_offsets64_double, LowerGSInfo::Gather(__gather_base_offsets64_double)},

        {__pseudo_scatter32_i8, LowerGSInfo::Scatter(__scatter32_generic_i8, __scatter32_i8)},
        {__pseudo_scatter32_i16, LowerGSInfo::Scatter(__scatter32_generic_i16, __scatter32_i16)},
        {__pseudo_scatter32_half, LowerGSInfo::Scatter(__scatter32_generic_half, __scatter32_half)},
        {__pseudo_scatter32_i32, LowerGSInfo::Scatter(__scatter32_generic_i32, __scatter32_i32)},
        {__pseudo_scatter32_float, LowerGSInfo::Scatter(__scatter32_generic_float, __scatter32_float)},
        {__pseudo_scatter32_i64, LowerGSInfo::Scatter(__scatter32_generic_i64, __scatter32_i64)},
        {__pseudo_scatter32_double, LowerGSInfo::Scatter(__scatter32_generic_double, __scatter32_double)},

        {__pseudo_scatter64_i8, LowerGSInfo::Scatter(__scatter64_generic_i8, __scatter64_i8)},
        {__pseudo_scatter64_i16, LowerGSInfo::Scatter(__scatter64_generic_i16, __scatter64_i16)},
        {__pseudo_scatter64_half, LowerGSInfo::Scatter(__scatter64_generic_half, __scatter64_half)},
        {__pseudo_scatter64_i32, LowerGSInfo::Scatter(__scatter64_generic_i32, __scatter64_i32)},
        {__pseudo_scatter64_float, LowerGSInfo::Scatter(__scatter64_generic_float, __scatter64_float)},
        {__pseudo_scatter64_i64, LowerGSInfo::Scatter(__scatter64_generic_i64, __scatter64_i64)},
        {__pseudo_scatter64_double, LowerGSInfo::Scatter(__scatter64_generic_double, __scatter64_double)},

        {__pseudo_scatter_factored_base_offsets32_i8, LowerGSInfo::Scatter(__scatter_factored_base_offsets32_i8)},
        {__pseudo_scatter_factored_base_offsets32_i16, LowerGSInfo::Scatter(__scatter_factored_base_offsets32_i16)},
        {__pseudo_scatter_factored_base_offsets32_half, LowerGSInfo::Scatter(__scatter_factored_base_offsets32_half)},
        {__pseudo_scatter_factored_base_offsets32_i32, LowerGSInfo::Scatter(__scatter_factored_base_offsets32_i32)},
        {__pseudo_scatter_factored_base_offsets32_float, LowerGSInfo::Scatter(__scatter_factored_base_offsets32_float)},
        {__pseudo_scatter_factored_base_offsets32_i64, LowerGSInfo::Scatter(__scatter_factored_base_offsets32_i64)},
        {__pseudo_scatter_factored_base_offsets32_double,
         LowerGSInfo::Scatter(__scatter_factored_base_offsets32_double)},

        {__pseudo_scatter_factored_base_offsets64_i8, LowerGSInfo::Scatter(__scatter_factored_base_offsets64_i8)},
        {__pseudo_scatter_factored_base_offsets64_i16, LowerGSInfo::Scatter(__scatter_factored_base_offsets64_i16)},
        {__pseudo_scatter_factored_base_offsets64_half, LowerGSInfo::Scatter(__scatter_factored_base_offsets64_half)},
        {__pseudo_scatter_factored_base_offsets64_i32, LowerGSInfo::Scatter(__scatter_factored_base_offsets64_i32)},
        {__pseudo_scatter_factored_base_offsets64_float, LowerGSInfo::Scatter(__scatter_factored_base_offsets64_float)},
        {__pseudo_scatter_factored_base_offsets64_i64, LowerGSInfo::Scatter(__scatter_factored_base_offsets64_i64)},
        {__pseudo_scatter_factored_base_offsets64_double,
         LowerGSInfo::Scatter(__scatter_factored_base_offsets64_double)},

        {__pseudo_scatter_base_offsets32_i8, LowerGSInfo::Scatter(__scatter_base_offsets32_i8)},
        {__pseudo_scatter_base_offsets32_i16, LowerGSInfo::Scatter(__scatter_base_offsets32_i16)},
        {__pseudo_scatter_base_offsets32_half, LowerGSInfo::Scatter(__scatter_base_offsets32_half)},
        {__pseudo_scatter_base_offsets32_i32, LowerGSInfo::Scatter(__scatter_base_offsets32_i32)},
        {__pseudo_scatter_base_offsets32_float, LowerGSInfo::Scatter(__scatter_base_offsets32_float)},
        {__pseudo_scatter_base_offsets32_i64, LowerGSInfo::Scatter(__scatter_base_offsets32_i64)},
        {__pseudo_scatter_base_offsets32_double, LowerGSInfo::Scatter(__scatter_base_offsets32_double)},

        {__pseudo_scatter_base_offsets64_i8, LowerGSInfo::Scatter(__scatter_base_offsets64_i8)},
        {__pseudo_scatter_base_offsets64_i16, LowerGSInfo::Scatter(__scatter_base_offsets64_i16)},
        {__pseudo_scatter_base_offsets64_half, LowerGSInfo::Scatter(__scatter_base_offsets64_half)},
        {__pseudo_scatter_base_offsets64_i32, LowerGSInfo::Scatter(__scatter_base_offsets64_i32)},
        {__pseudo_scatter_base_offsets64_float, LowerGSInfo::Scatter(__scatter_base_offsets64_float)},
        {__pseudo_scatter_base_offsets64_i64, LowerGSInfo::Scatter(__scatter_base_offsets64_i64)},
        {__pseudo_scatter_base_offsets64_double, LowerGSInfo::Scatter(__scatter_base_offsets64_double)},

        {__pseudo_prefetch_read_varying_1, LowerGSInfo::Prefetch(__prefetch_read_varying_1)},
        {__pseudo_prefetch_read_varying_1_native, LowerGSInfo::Prefetch(__prefetch_read_varying_1_native)},

        {__pseudo_prefetch_read_varying_2, LowerGSInfo::Prefetch(__prefetch_read_varying_2)},
        {__pseudo_prefetch_read_varying_2_native, LowerGSInfo::Prefetch(__prefetch_read_varying_2_native)},

        {__pseudo_prefetch_read_varying_3, LowerGSInfo::Prefetch(__prefetch_read_varying_3)},
        {__pseudo_prefetch_read_varying_3_native, LowerGSInfo::Prefetch(__prefetch_read_varying_3_native)},

        {__pseudo_prefetch_read_varying_nt, LowerGSInfo::Prefetch(__prefetch_read_varying_nt)},
        {__pseudo_prefetch_read_varying_nt_native, LowerGSInfo::Prefetch(__prefetch_read_varying_nt_native)},

        {__pseudo_prefetch_write_varying_1, LowerGSInfo::Prefetch(__prefetch_write_varying_1)},
        {__pseudo_prefetch_write_varying_1_native, LowerGSInfo::Prefetch(__prefetch_write_varying_1_native)},

        {__pseudo_prefetch_write_varying_2, LowerGSInfo::Prefetch(__prefetch_write_varying_2)},
        {__pseudo_prefetch_write_varying_2_native, LowerGSInfo::Prefetch(__prefetch_write_varying_2_native)},

        {__pseudo_prefetch_write_varying_3, LowerGSInfo::Prefetch(__prefetch_write_varying_3)},
        {__pseudo_prefetch_write_varying_3_native, LowerGSInfo::Prefetch(__prefetch_write_varying_3_native)},
    };

    llvm::Function *calledFunc = callInst->getCalledFunction();

    auto name = calledFunc->getName().str();
    auto it = replacementRules.find(name);
    if (it == replacementRules.end()) {
        // it is not a call of __pseudo function stored in replacementRules
        return false;
    }
    LowerGSInfo *info = &it->second;

    // Get the source position from the metadata attached to the call
    // instruction so that we can issue PerformanceWarning()s below.
    SourcePos pos;
    bool gotPosition = LLVMGetSourcePosFromMetadata(callInst, &pos);

    llvm::Module *M = callInst->getModule();
    callInst->setCalledFunction(info->actualFunc(M));
    // Check for alloca and if not alloca - generate __gather and change arguments
    if (gotPosition && (g->target->getVectorWidth() > 1) && (g->opt.level > 0)) {
        if (info->isGather())
            PerformanceWarning(pos, "Gather required to load value.");
        else if (!info->isPrefetch())
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
