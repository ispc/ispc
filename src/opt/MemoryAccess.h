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

/** How the common base of a General-form access was recognized. None of
    these establishes pointer provenance for LLVM alias analysis (#3983): a
    base recovered from the integer arithmetic is a candidate decomposition,
    not a proof that every lane's access is based on that pointer. */
enum class BaseOrigin : uint8_t {
    Pointer,       ///< a pointer-typed value
    PtrToInt,      ///< a ptrtoint instruction or constant expression
    LoadedInteger, ///< an integer load: the legacy heuristic for a pointer
                   ///< that LLVM loaded as an integer, kept for compatibility
};

/** Provenance of a varying address. RecoverBaseOffsets() always reports
    Unknown: marking an address as integer-built needs a frontend
    representation, and PointerDerived needs a proof of the pointer
    derivation; neither is available here. */
enum class AddressProvenance : uint8_t { Unknown, IntegerBuilt, PointerDerived };

/** A common scalar base plus per-lane offsets equal to the addresses of a
    General-form access: addresses == splat(casts(base)) + offsets at the
    address width, exactly as the existing matcher decomposed them. The
    recipe refers only to existing values and lane constants; nothing is
    created until MaterializeBaseOffsets(). */
struct BaseOffsetsRecipe {
    /** The common scalar base, before baseCasts. */
    llvm::Value *base = nullptr;
    BaseOrigin baseOrigin = BaseOrigin::Pointer;

    /** Scalar casts to replay on base, innermost first: the vector casts the
        addresses went through. */
    struct Cast {
        llvm::Instruction::CastOps opcode;
        llvm::Type *type;
    };
    llvm::SmallVector<Cast, 2> baseCasts;

    /** Vector type of the offsets: the type of the addresses. */
    llvm::FixedVectorType *offsetsType = nullptr;

    /** Per-lane constant offsets at the lane width when the innermost
        addresses are a constant vector of a global's address plus
        displacements; otherwise empty and the sum of terms starts at zero.
        The offsets have the addresses' type; the legacy matcher used the
        target's pointer width, which is the same for every call the frontend
        emits. */
    llvm::SmallVector<llvm::APInt, ISPC_MAX_NVEC> laneOffsets;

    /** Addends of the offsets, innermost first:
        offsets = (laneOffsets or 0) + terms[0] + terms[1] + ... */
    struct Term {
        /** An existing vector value, or the vector whose lane 0 is broadcast
            when splatAnchor is set. */
        llvm::Value *value = nullptr;
        /** When set, the term is a shuffle broadcasting lane 0 of value,
            inserted before (or after, if insertAfterAnchor) this instruction. */
        llvm::Instruction *splatAnchor = nullptr;
        bool insertAfterAnchor = false;
    };
    llvm::SmallVector<Term, 4> terms;

    AddressProvenance provenance = AddressProvenance::Unknown;
};

/** Read-only query: recovers a common base and per-lane offsets from the
    addresses of a General-form gather, scatter or prefetch. Recognizes the
    forms the existing matcher handled: a uniform value broadcast through an
    insert chain, a shuffle or a constant splat, possibly behind vector casts;
    adds (or disjoint ors) of such a base and other terms; a constant vector of
    one global's address plus displacements; and the aggregate insert/extract
    form. Returns no value for anything else, for unequal bases and when the
    budget of visited values is exhausted. Never inserts instructions, changes
    operands or use lists, or creates LLVM constant expressions. */
std::optional<BaseOffsetsRecipe> RecoverBaseOffsets(const MemoryAccess &access);

struct BaseOffsetsValues {
    llvm::Value *base = nullptr;    ///< scalar, after the recipe's casts
    llvm::Value *offsets = nullptr; ///< vector of the recipe's offsetsType
};

/** Creates the recipe's values, inserting new instructions before
    insertBefore (shuffles at their anchors), in the shape the existing
    matcher produced. */
BaseOffsetsValues MaterializeBaseOffsets(const BaseOffsetsRecipe &recipe, llvm::Instruction *insertBefore);

/** Prints the decoded memory accesses of a function to stderr without
    rewriting anything. Registered with ispc-opt for tests
    ("print-memory-access"); not a user-facing compiler option. */
struct MemoryAccessPrinterPass : public llvm::PassInfoMixin<MemoryAccessPrinterPass> {
    llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace ispc
