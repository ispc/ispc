/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "MemoryAccess.h"

namespace ispc {

namespace {

const char *lKindName(MemoryOpKind kind) {
    switch (kind) {
    case MemoryOpKind::Gather:
        return "gather";
    case MemoryOpKind::Scatter:
        return "scatter";
    case MemoryOpKind::Prefetch:
        return "prefetch";
    case MemoryOpKind::MaskedLoad:
        return "masked load";
    case MemoryOpKind::MaskedStore:
        return "masked store";
    }
    return "?";
}

const char *lFormName(MemoryOpForm form) {
    switch (form) {
    case MemoryOpForm::General:
        return "general";
    case MemoryOpForm::BaseOffsets:
        return "base-offsets";
    case MemoryOpForm::Factored:
        return "factored";
    case MemoryOpForm::Pointer:
        return "pointer";
    }
    return "?";
}

const char *lFlavorName(MemoryOpFlavor flavor) {
    switch (flavor) {
    case MemoryOpFlavor::Pseudo:
        return "pseudo";
    case MemoryOpFlavor::Lowered:
        return "lowered";
    case MemoryOpFlavor::Generic:
        return "generic";
    case MemoryOpFlavor::Blend:
        return "blend";
    }
    return "?";
}

const char *lMaskStatusName(MaskStatus status) {
    switch (status) {
    case MaskStatus::all_on:
        return "all on";
    case MaskStatus::all_off:
        return "all off";
    case MaskStatus::mixed:
        return "mixed";
    case MaskStatus::unknown:
        return "unknown";
    }
    return "?";
}

void lPrintOperand(llvm::raw_ostream &os, const char *role, llvm::Value *v) {
    if (v == nullptr) {
        return;
    }
    os << "  " << role << ": ";
    v->printAsOperand(os, /*PrintType=*/true);
    os << "\n";
}

} // namespace

unsigned MemoryAccess::laneCount() const {
    return llvm::cast<llvm::FixedVectorType>(mask()->getType())->getNumElements();
}

MaskStatus MemoryAccess::maskStatus() const {
    llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(mask());
    if (c == nullptr || c->containsUndefOrPoisonElement()) {
        return MaskStatus::unknown;
    }
    return GetMaskStatusFromValue(mask(), laneCount());
}

std::optional<MemoryAccess> DecodeMemoryAccess(llvm::CallInst *call) {
    // getCalledFunction() is null for an indirect call and for a call whose
    // function type differs from the callee's.
    llvm::Function *callee = call->getCalledFunction();
    if (callee == nullptr) {
        return std::nullopt;
    }
    const MemoryOp *op = LookupMemoryOp(callee->getName());
    if (op == nullptr || !HasMemoryOpSignature(*op, call->getFunctionType())) {
        return std::nullopt;
    }

    return MemoryAccess{call, op};
}

void MemoryAccess::print(llvm::raw_ostream &os) const {
    os << lKindName(kind()) << " " << op->name << ": " << lFormName(form()) << ", " << lFlavorName(flavor());
    if (element() != MemoryElement::None) {
        os << ", " << *elementType() << " (" << elementBytes() << " bytes)";
    }
    os << ", " << laneCount() << " lanes";
    if (offsetBits() != 0) {
        os << ", " << offsetBits() << "-bit " << (form() == MemoryOpForm::General ? "addresses" : "offsets");
    }
    os << "\n";
    if (!call->getType()->isVoidTy()) {
        lPrintOperand(os, "result", call);
    }
    lPrintOperand(os, "addresses", addresses());
    lPrintOperand(os, "base", base());
    lPrintOperand(os, "scale", scale());
    lPrintOperand(os, "offsets", offsets());
    lPrintOperand(os, "delta", delta());
    lPrintOperand(os, "value", value());
    os << "  mask: ";
    mask()->printAsOperand(os, /*PrintType=*/true);
    os << " (" << lMaskStatusName(maskStatus()) << ")\n";
    SourcePos pos;
    if (sourcePos(&pos)) {
        os << "  source: " << (pos.name ? pos.name : "?") << ":" << pos.first_line << ":" << pos.first_column << "\n";
    } else {
        os << "  source: unknown\n";
    }
}

llvm::PreservedAnalyses MemoryAccessPrinterPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::raw_ostream &os = llvm::errs();
    os << "memory accesses of @" << F.getName() << ":\n";
    for (llvm::BasicBlock &bb : F) {
        for (llvm::Instruction &inst : bb) {
            llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(&inst);
            if (call == nullptr) {
                continue;
            }
            std::optional<MemoryAccess> access = DecodeMemoryAccess(call);
            if (access) {
                access->print(os);
                continue;
            }
            // A direct call of a registered name that did not decode,
            // including one whose function type differs from the callee's.
            llvm::Function *callee = llvm::dyn_cast<llvm::Function>(call->getCalledOperand());
            if (callee != nullptr && LookupMemoryOp(callee->getName())) {
                os << "malformed " << callee->getName() << ": not decoded\n";
            }
        }
    }
    return llvm::PreservedAnalyses::all();
}

} // namespace ispc
