/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file MemoryPlan.cpp
    @brief The dump of the plans of the memory passes (--dump-memory-plans).
*/

#include "MemoryPlan.h"

#include <llvm/Support/raw_ostream.h>

namespace ispc {

bool DumpMemoryPlans() { return g->dumpMemoryPlans; }

std::string MemoryPlanAccessId(llvm::CallInst *call) {
    llvm::MDNode *md = call->getMetadata(MemoryAccessIdMetadata);
    llvm::MDString *id =
        md != nullptr && md->getNumOperands() == 1 ? llvm::dyn_cast<llvm::MDString>(md->getOperand(0)) : nullptr;
    return id != nullptr ? id->getString().str() : "?";
}

std::string MemoryPlanPosition(llvm::CallInst *call) {
    SourcePos pos;
    if (!LLVMGetSourcePosFromMetadata(call, &pos) || pos.name == nullptr) {
        return "?";
    }
    return std::string(pos.name) + ":" + std::to_string(pos.first_line) + ":" + std::to_string(pos.first_column);
}

void MemoryPlanDump::identify(llvm::CallInst *call) {
    std::string id = MemoryPlanAccessId(call);
    std::string stageName = stage > 0 ? std::to_string(stage) : "?";
    if (id == "?") {
        id = call->getFunction()->getName().str() + "@" + stageName + ":" + std::to_string(++newIds);
    } else if (ids.seen.count(id) != 0) {
        // A copy of an access seen before in this run, made by inlining or
        // unrolling: it gets an id of its own, which it keeps.
        id += "+" + stageName + ":" + std::to_string(++ids.copies[id]);
    } else {
        ids.seen.insert(id);
        return;
    }
    ids.seen.insert(id);
    llvm::LLVMContext &ctx = call->getContext();
    call->setMetadata(MemoryAccessIdMetadata, llvm::MDNode::get(ctx, llvm::MDString::get(ctx, id)));
}

std::optional<MemoryPlanDump> StartMemoryPlanDump(const char *pass, int stage, std::shared_ptr<MemoryPlanIds> &ids) {
    if (!DumpMemoryPlans()) {
        return std::nullopt;
    }
    if (ids == nullptr) {
        ids = std::make_shared<MemoryPlanIds>();
    }
    return MemoryPlanDump(pass, stage, *ids);
}

void MemoryPlanDump::printKeep(llvm::raw_ostream &os, llvm::CallInst *call, const char *reason) {
    identify(call);
    printPrefix(os, call, 0);
    os << "keep: " << reason << "\n";
}

void MemoryPlanDump::printPrefix(llvm::raw_ostream &os, llvm::CallInst *call, unsigned step) const {
    os << "memory-plan: " << pass << "@";
    if (stage > 0) {
        os << stage;
    } else {
        os << "?";
    }
    os << " " << call->getFunction()->getName() << " " << MemoryPlanPosition(call) << " " << MemoryPlanAccessId(call)
       << "#" << step;
    llvm::Function *callee = call->getCalledFunction();
    os << " " << (callee != nullptr ? callee->getName() : llvm::StringRef("?")) << " -> ";
}

static const char *lShapeName(PlanShape shape) {
    switch (shape) {
    case PlanShape::BasedForm:
        return "based";
    case PlanShape::Broadcast:
        return "broadcast";
    case PlanShape::Load:
        return "load";
    case PlanShape::ReverseLoad:
        return "reverse-load";
    case PlanShape::Store:
        return "store";
    case PlanShape::ReverseStore:
        return "reverse-store";
    case PlanShape::Unmasked:
        return "unmasked";
    case PlanShape::Removed:
        return "removed";
    }
    return "?";
}

static const char *lPointerName(PlanPointer pointer) {
    switch (pointer) {
    case PlanPointer::Lane0:
        return "lane0";
    case PlanPointer::FirstOnLane:
        return "first-on";
    case PlanPointer::LastLane:
        return "last";
    }
    return "?";
}

/** The lane width of an integer vector operand, as "i32" or "i64". */
static void lPrintLaneType(llvm::raw_ostream &os, llvm::Value *v) {
    if (v == nullptr) {
        os << "none";
        return;
    }
    v->getType()->getScalarType()->print(os);
}

void MemoryPlanDump::print(llvm::raw_ostream &os, const MemoryAccess &access, unsigned step,
                           const std::optional<MemoryPlan> &plan) const {
    printPrefix(os, access.call, step);
    if (!plan) {
        os << "keep\n";
        return;
    }
    os << plan->rule << ": " << lShapeName(plan->shape);
    if (plan->function != nullptr) {
        os << " " << plan->function->getName();
    }
    switch (plan->shape) {
    case PlanShape::BasedForm: {
        os << " scale=";
        if (llvm::ConstantInt *scale = llvm::dyn_cast_or_null<llvm::ConstantInt>(plan->scale)) {
            os << scale->getSExtValue();
        } else {
            os << "?";
        }
        os << " offsets=";
        lPrintLaneType(os, plan->offsets);
        if (plan->delta != nullptr) {
            os << " delta=" << (llvm::isa<llvm::Constant>(plan->delta) ? "const" : "expr");
        }
        break;
    }
    case PlanShape::Broadcast:
    case PlanShape::Load:
    case PlanShape::ReverseLoad:
    case PlanShape::Store:
    case PlanShape::ReverseStore:
        os << " offsets=";
        lPrintLaneType(os, plan->offsets);
        os << " pointer=" << lPointerName(plan->pointer);
        break;
    case PlanShape::Unmasked:
        os << " align=" << plan->align;
        break;
    case PlanShape::Removed:
        break;
    }
    os << "\n";
}

} // namespace ispc
