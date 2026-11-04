/*
  Copyright (c) 2022-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ReplacePseudoMemoryOps.h"
#include "MemoryOpRegistry.h"

namespace ispc {

/** Helper function to return a FixedVectorType if the type (after unwrapping arrays)
    is a vector, or nullptr otherwise.
*/
static llvm::FixedVectorType *lExtractVectorType(llvm::Type *type) {
    // First unwrap arrays
    llvm::ArrayType *at = nullptr;
    while ((at = llvm::dyn_cast<llvm::ArrayType>(type))) {
        type = at->getElementType();
    }

    // Return the type if it's a FixedVectorType, nullptr otherwise
    return llvm::dyn_cast<llvm::FixedVectorType>(type);
}

/** Helper function to recursively analyze struct field access at a given offset.
    Returns the FixedVectorType if found, or nullptr otherwise.
*/
static llvm::FixedVectorType *lCheckStructFieldAtOffset(llvm::StructType *st, uint64_t targetOffset,
                                                        const llvm::DataLayout &DL) {
    const llvm::StructLayout *SL = DL.getStructLayout(st);
    uint64_t structSize = SL->getSizeInBytes().getFixedValue();

    // Check bounds
    if (targetOffset >= structSize) {
        return nullptr;
    }

    // Find containing field
    unsigned fieldIndex = SL->getElementContainingOffset(targetOffset);
    if (fieldIndex >= st->getNumElements()) {
        return nullptr;
    }

    llvm::Type *fieldType = st->getElementType(fieldIndex);

    // If this field is a nested struct, recurse
    if (llvm::StructType *nestedSt = llvm::dyn_cast<llvm::StructType>(fieldType)) {
        uint64_t fieldOffset = SL->getElementOffset(fieldIndex).getFixedValue();
        uint64_t offsetWithinField = targetOffset - fieldOffset;
        return lCheckStructFieldAtOffset(nestedSt, offsetWithinField, DL);
    }

    // Leaf field - extract vector type
    return lExtractVectorType(fieldType);
}

/** This routine attempts to find the target vector type for blend operations
    on the given pointer. Returns the FixedVectorType if the pointer points to
    stack-allocated memory suitable for blending, or nullptr otherwise.
    It's conservative and may return nullptr for memory that could actually
    support blending. The basic strategy is to traverse through the operands
    and analyze the underlying allocated type.
*/
static llvm::FixedVectorType *lGetUnderlyingVectorType(llvm::Value *lvalue) {
    // Handle bitcast by recursing to operand
    if (llvm::BitCastInst *bc = llvm::dyn_cast<llvm::BitCastInst>(lvalue)) {
        return lGetUnderlyingVectorType(bc->getOperand(0));
    }

    // Handle direct alloca access
    if (llvm::AllocaInst *ai = llvm::dyn_cast<llvm::AllocaInst>(lvalue)) {
        llvm::Type *allocatedType = ai->getAllocatedType();

        if (llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(allocatedType)) {
            llvm::Module *M = ai->getModule();
            const llvm::DataLayout &DL = M->getDataLayout();
            return lCheckStructFieldAtOffset(st, 0, DL);
        } else {
            // For non-struct types, use the existing logic
            return lExtractVectorType(allocatedType);
        }
    }

    // Handle GEP instruction
    if (llvm::GetElementPtrInst *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(lvalue)) {
        // Must have constant indices for analysis
        if (!gep->hasAllConstantIndices()) {
            return lGetUnderlyingVectorType(gep->getOperand(0));
        }

        // Must be based on a struct alloca
        llvm::Value *basePtr = gep->getOperand(0);
        llvm::AllocaInst *ai = llvm::dyn_cast<llvm::AllocaInst>(basePtr);
        if (!ai) {
            return lGetUnderlyingVectorType(basePtr);
        }

        llvm::StructType *st = llvm::dyn_cast<llvm::StructType>(ai->getAllocatedType());
        if (!st) {
            return lGetUnderlyingVectorType(basePtr);
        }

        // Calculate offset and recursively analyze struct
        llvm::Module *M = ai->getModule();
        const llvm::DataLayout &DL = M->getDataLayout();
        llvm::APInt offset(DL.getIndexSizeInBits(gep->getPointerAddressSpace()), 0);

        if (!gep->accumulateConstantOffset(DL, offset)) {
            return nullptr;
        }

        uint64_t targetOffset = offset.getZExtValue();
        return lCheckStructFieldAtOffset(st, targetOffset, DL);
    }

    return nullptr;
}

static bool lReplacePseudoMaskedStore(llvm::CallInst *callInst) {
    const MemoryOp *op = LookupMemoryOp(callInst->getCalledFunction()->getName());
    if (op == nullptr || op->kind() != MemoryOpKind::MaskedStore || op->flavor() != MemoryOpFlavor::Pseudo) {
        return false;
    }

    llvm::Value *lvalue = callInst->getArgOperand(0);
    llvm::Value *rvalue = callInst->getArgOperand(1);
    llvm::Value *mask = callInst->getArgOperand(2);

    // We need to choose between doing the load + blend + store trick,
    // or serializing the masked store. Even on targets with a native
    // masked store instruction, this is preferable since it lets us
    // keep values in registers rather than going out to the stack.
    bool doBlend = false;
    if (!g->opt.disableBlendedMaskedStores) {
        if (llvm::FixedVectorType *vt = lGetUnderlyingVectorType(lvalue)) {
            auto &DL = callInst->getModule()->getDataLayout();
            doBlend = DL.getTypeSizeInBits(vt) >= DL.getTypeSizeInBits(rvalue->getType());
        }
    }

    // Generate the call to the appropriate masked store function and
    // replace the __pseudo_* one with it.
    llvm::Module *M = callInst->getModule();
    MemoryOpKey lowered = op->key;
    lowered.flavor = doBlend ? MemoryOpFlavor::Blend : MemoryOpFlavor::Lowered;
    llvm::Function *fms = GetMemoryOpFunction(M, lowered);
    llvm::Instruction *inst = LLVMCallInst(fms, lvalue, rvalue, mask, "", callInst);
    LLVMCopyMetadata(inst, callInst);

    callInst->eraseFromParent();
    return true;
}

static bool lReplacePseudoGS(llvm::CallInst *callInst) {
    const MemoryOp *op = LookupMemoryOp(callInst->getCalledFunction()->getName());
    if (op == nullptr || op->flavor() != MemoryOpFlavor::Pseudo ||
        (op->kind() != MemoryOpKind::Gather && op->kind() != MemoryOpKind::Scatter &&
         op->kind() != MemoryOpKind::Prefetch)) {
        return false;
    }
    const bool isGather = op->kind() == MemoryOpKind::Gather;
    const bool isPrefetch = op->kind() == MemoryOpKind::Prefetch;

    // The target builtin of the same form, or for a general gather/scatter
    // the generic one when the native instruction is disabled.
    MemoryOpKey lowered = op->key;
    lowered.flavor = MemoryOpFlavor::Lowered;
    if (op->form() == MemoryOpForm::General) {
        if ((isGather && g->target->hasGather() && g->opt.disableGathers) ||
            (op->kind() == MemoryOpKind::Scatter && g->target->hasScatter() && g->opt.disableScatters)) {
            lowered.flavor = MemoryOpFlavor::Generic;
        }
    }

    // Get the source position from the metadata attached to the call
    // instruction so that we can issue PerformanceWarning()s below.
    SourcePos pos;
    bool gotPosition = LLVMGetSourcePosFromMetadata(callInst, &pos);

    llvm::Module *M = callInst->getModule();
    callInst->setCalledFunction(GetMemoryOpFunction(M, lowered));
    // Check for alloca and if not alloca - generate __gather and change arguments
    if (gotPosition && (g->target->getVectorWidth() > 1) && (g->opt.level > 0)) {
        if (isGather) {
            PerformanceWarning(pos, "Gather required to load value.");
        } else if (!isPrefetch) {
            PerformanceWarning(pos, "Scatter required to store value.");
        }
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
        if (callInst == nullptr || callInst->getCalledFunction() == nullptr) {
            continue;
        }

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
