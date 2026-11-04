/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file MemoryOpRegistry.h
    @brief The gather, scatter, prefetch and masked load/store builtins of
    core.isph, each described by a structured key: what it does (kind), how
    it takes its address (form), its element type, which implementation it
    is (flavor) and the lane width of its address or offset operand.

    Passes that recognize or replace these builtins look them up here
    instead of keeping their own name tables: a replacement is the same key
    with a different form, width or flavor. The names themselves are the
    builtin:: constants of builtins-decl.h.
*/

#pragma once

#include <llvm/ADT/StringRef.h>

#include <cstdint>

namespace llvm {
class Function;
class FunctionType;
class Module;
class Type;
} // namespace llvm

namespace ispc {

enum class MemoryOpKind : uint8_t { Gather, Scatter, Prefetch, MaskedLoad, MaskedStore };

enum class MemoryOpForm : uint8_t {
    General,     ///< (addresses, [value,] mask): absolute varying addresses
    BaseOffsets, ///< (base, scale, offsets, [value,] mask)
    Factored,    ///< (base, offsets, scale, delta, [value,] mask)
    Pointer,     ///< (pointer, [value,] mask): masked load/store of consecutive lanes
};

enum class MemoryElement : uint8_t { None, I8, I16, Half, I32, Float, I64, Double };

enum class MemoryOpFlavor : uint8_t {
    Pseudo,  ///< __pseudo_* placeholder, lowered by ReplacePseudoMemoryOps
    Lowered, ///< the target builtin
    Generic, ///< the generic (emulated) builtin of a General-form gather/scatter
    Blend,   ///< the load/blend variant of a masked load or store
};

/** Cache level and intent of a prefetch; None for the other kinds. A
    prefetch has only the General form, (addresses, mask): no target has a
    vector prefetch instruction. */
enum class PrefetchHint : uint8_t { None, Read1, Read2, Read3, ReadNT, Write1, Write2, Write3 };

/** What identifies a memory builtin. */
struct MemoryOpKey {
    MemoryOpKind kind = MemoryOpKind::Gather;
    MemoryOpForm form = MemoryOpForm::General;
    MemoryElement element = MemoryElement::None;
    MemoryOpFlavor flavor = MemoryOpFlavor::Pseudo;
    /** Lane width in bits of the addresses or offsets operand; 0 for the
        Pointer form, which has no integer address operand. */
    uint8_t offsetBits = 0;
    PrefetchHint hint = PrefetchHint::None;
};

/** Operand positions of a form, -1 when the form has no such operand. */
struct MemoryOpLayout {
    int8_t addresses = -1;
    int8_t base = -1;
    int8_t scale = -1;
    int8_t offsets = -1;
    int8_t delta = -1;
    int8_t value = -1;
    int8_t mask = -1;
    uint8_t numArgs = 0;
};

/** One registered memory builtin. Names only: LLVM types and functions are
    resolved for the current module and target when needed, since the
    registry outlives a target during multi-target compilation. */
struct MemoryOp {
    const char *name = nullptr;
    MemoryOpKey key;
    MemoryOpLayout layout;

    MemoryOpKind kind() const { return key.kind; }
    MemoryOpForm form() const { return key.form; }
    MemoryElement element() const { return key.element; }
    MemoryOpFlavor flavor() const { return key.flavor; }
    unsigned offsetBits() const { return key.offsetBits; }
};

/** The registered builtin with this name, or nullptr. */
const MemoryOp *LookupMemoryOp(llvm::StringRef name);

/** The registered builtin with this key, or nullptr. */
const MemoryOp *FindMemoryOp(const MemoryOpKey &key);

/** The declaration in M of the builtin with this key; nullptr if no builtin
    has the key or M does not declare it. */
llvm::Function *GetMemoryOpFunction(llvm::Module *M, const MemoryOpKey &key);

/** Scalar LLVM type of an element for the current context; nullptr for None. */
llvm::Type *GetMemoryElementType(MemoryElement element);

/** Size of an element in bytes; 0 for None. */
unsigned GetMemoryElementBytes(MemoryElement element);

/** True when type is the signature of op for some lane count: the
    parameters its layout names, in order, with an integer vector mask of at
    most ISPC_MAX_NVEC lanes that gives the lane count of every other
    vector, address and offset lanes of op's offset width, a pointer base,
    an i32 scale, a delta of the offsets' type, and op's element type for
    the stored value or the result. Variadic types never match. */
bool HasMemoryOpSignature(const MemoryOp &op, llvm::FunctionType *type);

/** Reports a fatal error if M declares or defines a registered builtin
    with another signature, so that a change of a declaration in core.isph
    or of a definition in builtins/ that the registry does not follow fails
    every compilation instead of silently disabling the passes that look for
    the builtin. Called once the builtins are linked. */
void VerifyMemoryOpDeclarations(llvm::Module *M);

} // namespace ispc
