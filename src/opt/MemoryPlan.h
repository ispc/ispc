/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file MemoryPlan.h
    @brief What ImproveMemoryOps replaces one gather, scatter or masked
    load/store call with, as a value.

    The pass is split into rules that look at an access and return a plan,
    and an emitter that rewrites the call as the plan says. A rule doesn't
    replace the call or change its operands. Like the rules it was moved
    from, it may create the values a plan refers to (such as the full
    offsets of a based form) in front of the call; a rule that finds no
    plan can leave such values unused.
*/

#pragma once

#include "MemoryAccess.h"

namespace ispc {

/** The kind of code a plan replaces the access with. */
enum class PlanShape : uint8_t {
    BasedForm,    ///< a based pseudo gather or scatter: function, base, scale, offsets and delta
    Broadcast,    ///< one scalar load at the pointer, broadcast to every lane
    Load,         ///< a masked load of consecutive lanes at the pointer
    ReverseLoad,  ///< a masked load of consecutive lanes at the pointer, then the lanes reversed
    Store,        ///< a masked store of consecutive lanes at the pointer
    ReverseStore, ///< the lanes of the value reversed, then a masked store of consecutive lanes
    Unmasked,     ///< a load or store without a mask, aligned to align
    Removed,      ///< nothing: the store is dropped, the load's result is poison
};

/** The lane whose offset gives the pointer of a Broadcast, Load, ReverseLoad,
    Store or ReverseStore plan. */
enum class PlanPointer : uint8_t {
    Lane0,       ///< base + offsets[0]
    FirstOnLane, ///< base + offsets[i] - i * stride for the first lane i that is on, or for lane 0 if none is
    LastLane,    ///< base + offsets[W - 1]: the lowest address of a reverse plan
};

/** One plan for one access. The mask and the stored value are the access's. */
struct MemoryPlan {
    MemoryPlan(PlanShape shape, const char *rule) : shape(shape), rule(rule) {}

    PlanShape shape;
    /** The rule of ImproveMemoryOps that made the plan. */
    const char *rule;
    /** BasedForm: the based builtin to call. Load, ReverseLoad, Store,
        ReverseStore: the masked load or store builtin. Otherwise nullptr. */
    llvm::Function *function = nullptr;
    /** BasedForm: the base operand. Broadcast, Load, ReverseLoad, Store,
        ReverseStore: the i8 pointer that offsets are counted from.
        Unmasked: the pointer. */
    llvm::Value *base = nullptr;
    /** BasedForm: the scale operand. */
    llvm::Value *scale = nullptr;
    /** BasedForm: the offsets operand. Broadcast, Load, ReverseLoad, Store,
        ReverseStore: the byte offset of every lane from base. */
    llvm::Value *offsets = nullptr;
    /** BasedForm of a factored builtin: the delta operand. */
    llvm::Value *delta = nullptr;
    /** Broadcast, Load, ReverseLoad, Store, ReverseStore. */
    PlanPointer pointer = PlanPointer::Lane0;
    /** FirstOnLane: the distance in bytes between consecutive lanes. */
    int stride = 0;
    /** Unmasked: the alignment of the load or store. */
    unsigned align = 0;
};

} // namespace ispc
