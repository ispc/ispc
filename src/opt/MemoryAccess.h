/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file MemoryAccess.h
    @brief Read-only description of a gather, scatter, prefetch or masked
    load/store call, intended as the one view of these calls for the
    gather/scatter passes (ImproveMemoryOps, GatherCoalescePass). It covers
    every builtin of the memory-op registry (MemoryOpRegistry.h): the
    __pseudo_* calls those passes see, and the lowered and generic builtins
    that exist after ReplacePseudoMemoryOps.

    DecodeMemoryAccess() identifies a call by the callee's name in the
    registry and verifies the call's signature against the registered one
    (HasMemoryOpSignature()), so operand roles come from the callee, not
    from the target's preferred implementation (useGather(), useScatter());
    that preference only decides what a pass rewrites a call into. The view
    only refers to existing values: it inserts no instructions, changes no
    operands or use lists, adds no declarations and copies no metadata.

    A layout is not a complete arithmetic specification. The lowering of a
    32-bit offset form on a 64-bit target decides how its i32 operands are
    extended: __gather_elt32_* in builtins/util.m4 computes
    base + sext64(offsets) * scale + sext64(delta), whereas the factored
    implementations in builtins/generic.ispc compute
    base + (offsets * scale + delta) at the operand width.
    These differ at overflow boundaries. The view records the operands and
    their widths; how a form combines them is the contract of that form's
    lowering and is not declared equivalent to another form's here.
*/

#pragma once

#include "ISPCPass.h"
#include "MemoryOpRegistry.h"

#include <optional>

namespace ispc {

/** A decoded memory call. The operand accessors read the call's current
    operands through the registered layout; the view does not interpret the
    address arithmetic. An operand that a form does not have is nullptr. */
struct MemoryAccess {
    llvm::CallInst *call = nullptr;
    const MemoryOp *op = nullptr; ///< the registered builtin, with its key and operand layout

    /** General: absolute addresses, one per lane. */
    llvm::Value *addresses() const { return operand(op->layout.addresses); }
    /** BaseOffsets, Factored: uniform base; Pointer: the pointer. */
    llvm::Value *base() const { return operand(op->layout.base); }
    /** BaseOffsets, Factored: uniform i32 scale of the offsets. */
    llvm::Value *scale() const { return operand(op->layout.scale); }
    /** BaseOffsets, Factored: varying offsets, offsetBits() wide. */
    llvm::Value *offsets() const { return operand(op->layout.offsets); }
    /** Factored: varying displacement, offsetBits() wide. */
    llvm::Value *delta() const { return operand(op->layout.delta); }
    /** Scatter, MaskedStore: stored value. */
    llvm::Value *value() const { return operand(op->layout.value); }
    /** The execution mask: integer lanes, any width. */
    llvm::Value *mask() const { return operand(op->layout.mask); }

    MemoryOpKind kind() const { return op->kind(); }
    MemoryOpForm form() const { return op->form(); }
    MemoryElement element() const { return op->element(); }
    MemoryOpFlavor flavor() const { return op->flavor(); }
    bool isPseudo() const { return op->flavor() == MemoryOpFlavor::Pseudo; }

    /** Scalar LLVM type of one element, resolved for the current context;
        nullptr for a prefetch. */
    llvm::Type *elementType() const { return GetMemoryElementType(element()); }
    /** Size of one element in bytes; 0 for a prefetch. */
    unsigned elementBytes() const { return GetMemoryElementBytes(element()); }
    /** Number of lanes, from the mask. */
    unsigned laneCount() const;
    /** Lane width in bits of addresses/offsets; 0 for the Pointer form. */
    unsigned offsetBits() const { return op->offsetBits(); }
    /** Status of the mask proved by its constant lanes: all on, all off,
        mixed, or unknown. A constant with undef or poison lanes is unknown:
        choosing values for those lanes is a refinement that the existing
        emitters make for themselves with GetMaskStatusFromValue(), not a fact
        about the original mask. */
    MaskStatus maskStatus() const;
    /** The element alignment the existing emitters assume for a replacement
        load/store: the element size; 0 for a prefetch. The call itself
        establishes nothing stronger; a recovered common base proves neither
        alignment nor dereferenceability. */
    unsigned naturalAlignment() const { return elementBytes(); }

    /** ISPC source position from the call's metadata; false if absent. */
    bool sourcePos(SourcePos *pos) const { return LLVMGetSourcePosFromMetadata(call, pos); }
    /** LLVM debug location of the call; may be empty. */
    llvm::DebugLoc debugLoc() const { return call->getDebugLoc(); }

    void print(llvm::raw_ostream &os) const;

  private:
    llvm::Value *operand(int8_t index) const { return index < 0 ? nullptr : call->getArgOperand(index); }
};

/** Decodes call if its callee is a registered memory builtin and the call
    has the registered signature (HasMemoryOpSignature()). Returns no value
    for other calls, for indirect calls, for calls whose function type
    differs from the callee's and for malformed forms, including variadic
    declarations and more than ISPC_MAX_NVEC lanes, without accessing
    operands under an assumed layout. */
std::optional<MemoryAccess> DecodeMemoryAccess(llvm::CallInst *call);

/** Prints the decoded memory accesses of a function to stderr without
    rewriting anything. Registered with ispc-opt for tests
    ("print-memory-access"); not a user-facing compiler option. */
struct MemoryAccessPrinterPass : public llvm::PassInfoMixin<MemoryAccessPrinterPass> {
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace ispc
