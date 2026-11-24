/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file AddressDecoder.h
    @brief Reads a varying address, a vector of pointers, as the IR states it:
    a base, the object the lanes point into, where the base comes from, one
    term per variable GEP index and the constant byte offset of each lane.

    The decoder only reads what the IR says exactly. Each GEP index is a term
    with its own scale, so a constant index or an add at the index width is
    split off exactly, whereas nothing is looked through an extension: the
    index under it is reported with the extension instead. A form the
    decoder doesn't know is reported as unknown, never guessed.

    The decoder inserts no instructions and changes nothing. It reads the
    element type spellings of every supported LLVM version (`float`, `%S` and
    `[N x i8]`), multi-index struct GEPs, constant-expression GEPs of globals,
    constant vectors of pointers into one global and splats of a scalar
    pointer.
*/

#pragma once

#include "ISPCPass.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

#include <optional>

namespace ispc {

/** How the lanes of a varying address relate to one base. */
enum class AddressBase : uint8_t {
    Uniform, ///< one scalar pointer for every lane
    PerLane, ///< a vector of pointers with no common scalar: loaded, an argument, a select, a call
    Integer, ///< built from an integer with inttoptr; the integer recovery reads it
};

/** Where the innermost pointer value, under the GEPs, comes from. */
enum class AddressOrigin : uint8_t {
    Scalar,    ///< a scalar pointer: the pointer operand of a vector GEP, a splat, or a constant vector of
               ///< pointers into one global
    IntToPtr,  ///< an inttoptr of an integer vector
    Load,      ///< a loaded vector of pointers
    Argument,  ///< a function argument
    PhiSelect, ///< a phi or a select of vectors of pointers
    Call,      ///< the result of a call
    Other,     ///< anything else, like an insertelement chain of different pointers
};

/** One variable GEP index: the address contains index * coefficient at the
    index width of the address space. */
struct AddressTerm {
    /** The index, at the index width: a vector, or a scalar for a uniform
        index (a scalar GEP index or a splat). */
    llvm::Value *index = nullptr;
    /** The index is the same value in every lane. */
    bool uniform = false;
    /** How the index was extended to the index width. index ==
        ext(narrow) when extension isn't None. A GEP index narrower than the
        index width is sign extended by the GEP itself; then index is that
        operand and narrow equals it. */
    enum class Extension : uint8_t { None, Sign, Zero, ZeroNonNeg } extension = Extension::None;
    llvm::Value *narrow = nullptr;
    /** Bytes per unit of the index, at the index width: the stride of the
        GEP's indexed type, times the constant of a mul or shl at the index
        width that the decoder looked through. */
    llvm::APInt coefficient;
};

/** A decoded varying address. Exact: every lane's address is
    base + sum over terms of (index * coefficient) + laneOffsets[lane],
    computed at the index width, which wraps at the pointer width like an
    unflagged GEP. */
struct AddressDecode {
    AddressBase base = AddressBase::PerLane;
    AddressOrigin origin = AddressOrigin::Other;
    /** Uniform: the scalar pointer. PerLane and Integer: the innermost
        vector, under the GEPs (for Integer, the inttoptr). */
    llvm::Value *root = nullptr;
    /** The address is computed from the root by at least one GEP. */
    bool viaGEP = false;
    /** Every lane points into the object the uniform base points into: all
        of them were computed by GEPs from it. Then a lane that is off may be
        given any address in that object. */
    bool oneObject = false;
    /** The size in bytes of that object when it is known (an alloca of a
        constant size or a global variable) together with the constant offset
        of the root in it; 0 otherwise. */
    uint64_t objectBytes = 0;
    /** With objectBytes: the byte offset of the root in the object. */
    int64_t rootOffset = 0;
    llvm::SmallVector<AddressTerm, 4> terms;
    /** The constant byte offset of each lane, at the index width. */
    llvm::SmallVector<llvm::APInt, ISPC_MAX_NVEC> laneOffsets;
    /** The index width of the address space, in bits. */
    unsigned indexBits = 0;

    unsigned laneCount() const { return laneOffsets.size(); }
    /** True when every term is uniform: the lanes differ only by their
        constant offsets. */
    bool onlyUniformTerms() const;
    /** True when the constant offsets are the same in every lane. */
    bool laneOffsetsEqual() const;

    void print(llvm::raw_ostream &os) const;
};

/** Decodes a varying address, a fixed vector of pointers. Returns no value
    for another type, for a constant with undef or poison lanes and for an
    index that LLVM can't give a lane value for. */
std::optional<AddressDecode> DecodeVaryingAddress(llvm::Value *ptrs, const llvm::DataLayout &DL);

/** The vector of pointers that the integer addresses of a General access
    were converted from (a ptrtoint to the pointer width), or the addresses
    themselves if they are a vector of pointers; nullptr otherwise. */
llvm::Value *GetPointerAddresses(llvm::Value *addresses, const llvm::DataLayout &DL);

} // namespace ispc
