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
    from, it may create the values a plan refers to (a materialized base,
    offsets with the scale divided out, the full offsets of a based form)
    in front of the call; a rule that finds no plan can leave such values
    unused.
*/

#pragma once

#include "MemoryAccess.h"

#include <llvm/Support/raw_ostream.h>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

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
    /** BasedForm: the registered builtin of function, whose layout places
        the operands. Otherwise nullptr. */
    const MemoryOp *op = nullptr;
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

/** True when --dump-memory-plans asks for every decision of the memory
    passes on stderr, one line each. */
bool DumpMemoryPlans();

/** The id of the access of call as "<function>:<n>", or "?" without one. */
std::string MemoryPlanAccessId(llvm::CallInst *call);

/** The source position of call as "file:line:col", or "?". */
std::string MemoryPlanPosition(llvm::CallInst *call);

/** The ids that one run of a pass has seen in the functions of a module. */
struct MemoryPlanIds {
    std::set<std::string> seen;
    std::map<std::string, unsigned> copies;
};

/** The dump of one run of a pass on one function. A line starts with

      memory-plan: <pass>@<stage> <function> <position> <id>#<step> <callee> ->

    The stage is the pass's number in the optimization pipeline (as for
    --debug-phase; ? outside of it). The position is that of the call, or ?
    if it has none. The id is that of the access (MemoryAccessIdMetadata),
    which the calls that replace an access keep. The frontend's id
    <function>:<n> is lost, like the position, when LLVM merges two calls,
    for example the copies of a function body for an all on and a mixed
    mask; such a call gets the id <function>@<stage>:<n> from the first run
    that prints it. A copy of an access made by inlining or unrolling gets
    the id <id>+<stage>:<k> from the first run that sees a second call with
    the same id, in any function of the module. So every access printed has
    an id of its own in the module, and it keeps it in the later runs. The
    step counts the plans of one access in a run: a plan whose replacement
    is another call is followed by the plan of that call, with the next
    step. */
class MemoryPlanDump {
  public:
    MemoryPlanDump(const char *pass, int stage, MemoryPlanIds &ids) : pass(pass), stage(stage), ids(ids) {}

    /** Gives call an id of its own if it has none or shares its id with
        another call of this run. Called once for each access, before its
        first line. */
    void identify(llvm::CallInst *call);
    /** Prints the start of a line, up to and including "-> ". */
    void printPrefix(llvm::raw_ostream &os, llvm::CallInst *call, unsigned step) const;
    /** Identifies call and prints a line saying that it is kept, for reason. */
    void printKeep(llvm::raw_ostream &os, llvm::CallInst *call, const char *reason);
    /** Prints a line for the access with the plan that the rules of
        ImproveMemoryOps returned, or "keep" if they returned none. Called
        before the plan is emitted, while the access's call still exists. */
    void print(llvm::raw_ostream &os, const MemoryAccess &access, unsigned step,
               const std::optional<MemoryPlan> &plan) const;

  private:
    const char *pass;
    int stage;
    MemoryPlanIds &ids;
    unsigned newIds = 0;
};

/** The dump of one run of pass on one function with --dump-memory-plans, or
    no value without it. ids holds what the pass object has seen in the
    module; it is created on first use. */
std::optional<MemoryPlanDump> StartMemoryPlanDump(const char *pass, int stage, std::shared_ptr<MemoryPlanIds> &ids);

} // namespace ispc
