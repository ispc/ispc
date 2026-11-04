/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "MemoryAccess.h"

#include <llvm/Analysis/ValueTracking.h>

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

namespace {

/** How many values one query visits at most, over adds, casts, aggregate
    inserts and the scalar pointer check together. The legacy matcher had no
    bound and could revisit a shared subexpression exponentially often; real
    address expressions are a few dozen values at most, and anything beyond
    the budget is reported as unknown. */
constexpr int kRecoveryBudget = 256;

/** Given the two operands of a constant add expression, returns the
    ptrtoint base if op0 is one and op1 an integer constant, the delta. */
llvm::Constant *lGetConstantAddExprBaseOffset(llvm::Constant *op0, llvm::Constant *op1, llvm::ConstantInt **delta) {
    llvm::ConstantExpr *op = llvm::dyn_cast<llvm::ConstantExpr>(op0);
    if (op == nullptr || op->getOpcode() != llvm::Instruction::PtrToInt) {
        return nullptr;
    }
    llvm::ConstantInt *opDelta = llvm::dyn_cast<llvm::ConstantInt>(op1);
    if (opDelta == nullptr) {
        return nullptr;
    }
    *delta = opDelta;
    return op0;
}

bool lIsAddLike(llvm::BinaryOperator *bop) {
    return bop->getOpcode() == llvm::Instruction::Add || IsOrEquivalentToAdd(bop);
}

bool lIsVectorOfLanes(llvm::Type *t, unsigned lanes) {
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(t);
    return vt != nullptr && vt->getNumElements() == lanes;
}

/** One recovery query. Each method fills recipe only on success. */
struct BaseOffsetsRecovery {
    /** Lanes of the access: every vector on the way to the base must have
        this many, as the legacy matcher required through the target width. */
    unsigned lanes;
    int budget = kRecoveryBudget;

    explicit BaseOffsetsRecovery(unsigned lanes) : lanes(lanes) {}

    /** Accounts for visiting one value; false once the budget is exhausted. */
    bool visit() { return --budget >= 0; }

    /** Is v a pointer in the end? Given vec(offset) + ptrtoint(ptr), the base
        must not be vec(offset) with ptrtoint(ptr) treated as the offset: that
        would make GEPs like "gep inttoptr 8, i64 %ptr", which LLVM's alias
        analysis handles incorrectly since it assumes that only the first GEP
        operand is a pointer. A loaded integer is accepted as a pointer that
        LLVM turned from a load of a pointer with a ptrtoint into a load of
        i64; this heuristic is kept as it was, and recorded as its own origin. */
    std::optional<BaseOrigin> checkForActualPointer(llvm::Value *v) {
        while (v != nullptr && visit()) {
            if (llvm::isa<llvm::PointerType>(v->getType())) {
                return BaseOrigin::Pointer;
            }
            if (llvm::isa<llvm::PtrToIntInst>(v)) {
                return BaseOrigin::PtrToInt;
            }
            if (llvm::isa<llvm::LoadInst>(v)) {
                return BaseOrigin::LoadedInteger;
            }
            if (llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v)) {
                v = ci->getOperand(0);
                continue;
            }
            llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(v);
            if (ce != nullptr && ce->getOpcode() == llvm::Instruction::PtrToInt) {
                return BaseOrigin::PtrToInt;
            }
            return std::nullopt;
        }
        return std::nullopt;
    }

    /** The value inserted at index into the aggregate built by a chain of
        insertvalue instructions, or nullptr. */
    llvm::Value *extractFromInserts(llvm::Value *v, unsigned index) {
        while (visit()) {
            llvm::InsertValueInst *iv = llvm::dyn_cast<llvm::InsertValueInst>(v);
            if (iv == nullptr || iv->getNumIndices() != 1) {
                return nullptr;
            }
            if (iv->getIndices()[0] == index) {
                return iv->getInsertedValueOperand();
            }
            v = iv->getAggregateOperand();
        }
        return nullptr;
    }

    /** Do all lanes of v hold the same value that is a pointer in the end?
        With broadcastDetected the first non-undef lane of a lane-0 broadcast
        is used. On success sets recipe.base, baseOrigin and baseCasts. */
    bool commonBase(llvm::Value *v, bool broadcastDetected, BaseOffsetsRecipe &recipe) {
        if (!visit() || !lIsVectorOfLanes(v->getType(), lanes)) {
            return false;
        }
        if (llvm::isa<llvm::InsertElementInst>(v) || llvm::isa<llvm::ShuffleVectorInst>(v)) {
            // The insert chain LLVMFlattenInsertChain() walks counts against
            // the budget. (Its structural comparison of the inserted lanes is
            // shared with other users of the helper and not budgeted.)
            for (llvm::InsertElementInst *ie = llvm::dyn_cast<llvm::InsertElementInst>(v); ie != nullptr;
                 ie = llvm::dyn_cast<llvm::InsertElementInst>(ie->getOperand(0))) {
                if (!visit()) {
                    return false;
                }
            }
            // TODO: it is probably ok to allow undefined or poison elements
            // and return the base pointer if all other elements are equal.
            llvm::Value *element =
                LLVMFlattenInsertChain(v, lanes, /*compare=*/true, /*undef=*/false, broadcastDetected);
            std::optional<BaseOrigin> origin = checkForActualPointer(element);
            if (!origin) {
                return false;
            }
            recipe.base = element;
            recipe.baseOrigin = *origin;
            return true;
        }
        // Global and static arrays give constant splats.
        if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v)) {
            return constantBase(cv->getSplatValue(), recipe);
        }
        if (llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v)) {
            return constantBase(cdv->getSplatValue(), recipe);
        }
        // Pointers cast to integers of another width are handled by replaying
        // the vector cast on the scalar base. A cast between vectors of
        // different lane counts has no scalar counterpart.
        if (llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v)) {
            if (!lIsVectorOfLanes(ci->getOperand(0)->getType(), lanes) ||
                !commonBase(ci->getOperand(0), broadcastDetected, recipe)) {
                return false;
            }
            recipe.baseCasts.push_back({ci->getOpcode(), ci->getType()->getScalarType()});
            return true;
        }
        return false;
    }

    bool constantBase(llvm::Constant *splat, BaseOffsetsRecipe &recipe) {
        std::optional<BaseOrigin> origin = checkForActualPointer(splat);
        if (!origin) {
            return false;
        }
        recipe.base = splat;
        recipe.baseOrigin = *origin;
        return true;
    }

    /** Is ptrs a common base plus per-lane offsets? */
    bool recover(llvm::Value *ptrs, BaseOffsetsRecipe &recipe) {
        if (!visit()) {
            return false;
        }
        // %gep_offset = shufflevector <8 x i64> %0, <8 x i64> poison, <8 x i32> zeroinitializer
        llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(ptrs);
        bool broadcastDetected = shuffle != nullptr && LLVMIsValueUndef(shuffle->getOperand(1)) &&
                                 llvm::isa<llvm::ConstantAggregateZero>(shuffle->getShuffleMaskForBitcode());

        BaseOffsetsRecipe candidate;
        if (commonBase(ptrs, broadcastDetected, candidate)) {
            // A varying pointer with no indexing whose lanes are all the same
            // value: offsets of zero.
            if (broadcastDetected && !broadcastOffset(shuffle, candidate)) {
                return false;
            }
            recipe = std::move(candidate);
            return true;
        }

        llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(ptrs);
        if (bop != nullptr && lIsAddLike(bop)) {
            // A common base plus something: the other operand is a term of the
            // offsets.
            for (int i = 0; i < 2; ++i) {
                if (recover(bop->getOperand(i), candidate)) {
                    candidate.terms.push_back({bop->getOperand(1 - i)});
                    recipe = std::move(candidate);
                    return true;
                }
            }
            return false;
        }

        if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(ptrs)) {
            return constantLanes(cv, recipe);
        }

        llvm::ExtractValueInst *ev = llvm::dyn_cast<llvm::ExtractValueInst>(ptrs);
        if (ev != nullptr && ev->getNumIndices() == 1) {
            llvm::Value *inserted = extractFromInserts(ev->getAggregateOperand(), ev->getIndices()[0]);
            return inserted != nullptr && recover(inserted, recipe);
        }
        return false;
    }

    /** The offset a broadcast carries in its lane 0: either the lane-0
        constant of <c, poison, ...> added before the shuffle, or lane 0 of
        another binary operator that computes a part of the offset. */
    bool broadcastOffset(llvm::ShuffleVectorInst *shuffle, BaseOffsetsRecipe &recipe) {
        llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(shuffle->getOperand(0));
        if (bop == nullptr || !lIsAddLike(bop)) {
            return true;
        }
        if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(bop->getOperand(1))) {
            recipe.terms.push_back({cv, bop, /*insertAfterAnchor=*/false});
            return true;
        }
        // %another_bop = bop <16 x i32> %vec, <i32 7, i32 poison, ...>
        // %offsets = add <16 x i32> %another_bop, %base
        llvm::BinaryOperator *part = llvm::dyn_cast<llvm::BinaryOperator>(bop->getOperand(0));
        if (part == nullptr || !llvm::isa<llvm::FixedVectorType>(part->getType())) {
            // Base + offset pattern was not recognized.
            return false;
        }
        recipe.terms.push_back({part, part, /*insertAfterAnchor=*/true});
        return true;
    }

    /** Indexing into global arrays can give a constant vector whose lanes are
        ptrtoint of one global, or that plus a constant: one base and
        per-lane constant offsets. */
    bool constantLanes(llvm::ConstantVector *cv, BaseOffsetsRecipe &recipe) {
        llvm::FixedVectorType *vt = llvm::cast<llvm::FixedVectorType>(cv->getType());
        if (vt->getNumElements() != lanes || !vt->getElementType()->isIntegerTy()) {
            return false;
        }
        unsigned bits = vt->getElementType()->getIntegerBitWidth();
        BaseOffsetsRecipe candidate;
        for (unsigned i = 0; i < vt->getNumElements(); ++i) {
            llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(cv->getOperand(i));
            if (ce == nullptr) {
                return false;
            }
            llvm::Constant *elementBase = nullptr;
            llvm::ConstantInt *delta = nullptr;
            if (ce->getOpcode() == llvm::Instruction::PtrToInt) {
                elementBase = ce;
            } else if (ce->getOpcode() == llvm::Instruction::Add) {
                elementBase = lGetConstantAddExprBaseOffset(ce->getOperand(0), ce->getOperand(1), &delta);
                if (elementBase == nullptr) {
                    elementBase = lGetConstantAddExprBaseOffset(ce->getOperand(1), ce->getOperand(0), &delta);
                }
            }
            if (elementBase == nullptr) {
                return false;
            }
            if (candidate.base == nullptr) {
                candidate.base = elementBase;
                candidate.baseOrigin = BaseOrigin::PtrToInt;
            } else if (candidate.base != elementBase) {
                // Different program instances have different base pointers.
                return false;
            }
            candidate.laneOffsets.push_back(delta != nullptr ? delta->getValue() : llvm::APInt(bits, 0));
        }
        recipe = std::move(candidate);
        return true;
    }
};

const char *lBaseOriginName(BaseOrigin origin) {
    switch (origin) {
    case BaseOrigin::Pointer:
        return "pointer";
    case BaseOrigin::PtrToInt:
        return "ptrtoint";
    case BaseOrigin::LoadedInteger:
        return "loaded integer";
    }
    return "?";
}

const char *lProvenanceName(AddressProvenance provenance) {
    switch (provenance) {
    case AddressProvenance::Unknown:
        return "unknown";
    case AddressProvenance::IntegerBuilt:
        return "integer-built";
    case AddressProvenance::PointerDerived:
        return "pointer-derived";
    }
    return "?";
}

void lPrintRecipe(llvm::raw_ostream &os, const BaseOffsetsRecipe &recipe) {
    os << "  common base: ";
    recipe.base->printAsOperand(os, /*PrintType=*/true);
    os << " (" << lBaseOriginName(recipe.baseOrigin) << ")";
    for (const BaseOffsetsRecipe::Cast &cast : recipe.baseCasts) {
        os << ", " << llvm::Instruction::getOpcodeName(cast.opcode) << " to " << *cast.type;
    }
    os << "\n  offsets: ";
    if (!recipe.laneOffsets.empty()) {
        os << "lanes";
        for (const llvm::APInt &lane : recipe.laneOffsets) {
            os << " " << lane;
        }
        os << " (i" << recipe.laneOffsets[0].getBitWidth() << ")";
    } else {
        os << "0";
    }
    for (const BaseOffsetsRecipe::Term &term : recipe.terms) {
        os << " + " << (term.splatAnchor != nullptr ? "lane 0 of " : "");
        term.value->printAsOperand(os, /*PrintType=*/true);
    }
    os << "\n  provenance: " << lProvenanceName(recipe.provenance) << "\n";
}

} // namespace

std::optional<BaseOffsetsRecipe> RecoverBaseOffsets(const MemoryAccess &access) {
    if (access.form() != MemoryOpForm::General) {
        return std::nullopt;
    }
    if (g->debugPrint) {
        fprintf(stderr, "RecoverBaseOffsets\n");
        LLVMDumpValue(access.addresses());
    }
    BaseOffsetsRecipe recipe;
    BaseOffsetsRecovery recovery(access.laneCount());
    if (!recovery.recover(access.addresses(), recipe)) {
        return std::nullopt;
    }
    recipe.offsetsType = llvm::cast<llvm::FixedVectorType>(access.addresses()->getType());
    return recipe;
}

BaseOffsetsValues MaterializeBaseOffsets(const BaseOffsetsRecipe &recipe, llvm::Instruction *insertBefore) {
    BaseOffsetsValues values;
    values.base = recipe.base;
    for (const BaseOffsetsRecipe::Cast &cast : recipe.baseCasts) {
        values.base =
            llvm::CastInst::Create(cast.opcode, values.base, cast.type, llvm::Twine(values.base->getName()) + "_cast",
                                   ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
    }

    if (recipe.laneOffsets.empty()) {
        values.offsets = llvm::Constant::getNullValue(recipe.offsetsType);
    } else {
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> lanes;
        for (const llvm::APInt &lane : recipe.laneOffsets) {
            lanes.push_back(llvm::ConstantInt::get(recipe.offsetsType->getElementType(), lane));
        }
        values.offsets = llvm::ConstantVector::get(lanes);
    }
    for (const BaseOffsetsRecipe::Term &term : recipe.terms) {
        llvm::Value *addend = term.value;
        if (term.splatAnchor != nullptr) {
            llvm::FixedVectorType *vt = llvm::cast<llvm::FixedVectorType>(addend->getType());
            llvm::Value *zeroMask =
                llvm::ConstantVector::getSplat(llvm::ElementCount::get(vt->getNumElements(), false),
                                               llvm::Constant::getNullValue(llvm::Type::getInt32Ty(vt->getContext())));
            llvm::ShuffleVectorInst *shuffle = nullptr;
            if (term.insertAfterAnchor) {
                shuffle = new llvm::ShuffleVectorInst(addend, llvm::PoisonValue::get(vt), zeroMask, "shuffle");
                shuffle->insertAfter(term.splatAnchor);
            } else {
                shuffle = new llvm::ShuffleVectorInst(addend, llvm::PoisonValue::get(vt), zeroMask, "shuffle",
                                                      ISPC_INSERTION_POINT_INSTRUCTION(term.splatAnchor));
            }
            addend = shuffle;
        }
        values.offsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, values.offsets, addend, "new_offsets",
                                                      ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
    }
    return values;
}

namespace {

/** The lanes of a constant integer vector without undef or poison lanes. */
bool lConstantLanes(llvm::Value *v, llvm::SmallVectorImpl<llvm::APInt> &lanes) {
    llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(v);
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
    if (c == nullptr || vt == nullptr || !vt->getElementType()->isIntegerTy() || c->containsUndefOrPoisonElement()) {
        return false;
    }
    for (unsigned i = 0; i < vt->getNumElements(); ++i) {
        llvm::ConstantInt *ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(c->getAggregateElement(i));
        if (ci == nullptr) {
            return false;
        }
        lanes.push_back(ci->getValue());
    }
    return true;
}

/** Narrows constant lanes to 32 bits when each is in [INT32_MIN, INT32_MAX]. */
bool lNarrowLanes(const llvm::SmallVectorImpl<llvm::APInt> &lanes, llvm::SmallVectorImpl<llvm::APInt> &narrow) {
    for (const llvm::APInt &lane : lanes) {
        if (!lane.isSignedIntN(32)) {
            return false;
        }
        narrow.push_back(lane.sextOrTrunc(32));
    }
    return true;
}

void lPrintLanes(llvm::raw_ostream &os, const llvm::SmallVectorImpl<llvm::APInt> &lanes) {
    os << "lanes";
    for (const llvm::APInt &lane : lanes) {
        os << " " << lane;
    }
}

/** How the lowering of a base-offset or factored access combines its
    operands, or no value when no lowering is written for its index width
    and the pointer width of its base. */
std::optional<OffsetArithmetic> lOffsetArithmetic(const MemoryAccess &access) {
    unsigned pointerBits = access.call->getModule()->getDataLayout().getPointerSizeInBits(
        access.base()->getType()->getPointerAddressSpace());
    if (access.offsetBits() == pointerBits) {
        return OffsetArithmetic::AddressWidth;
    }
    if (access.offsetBits() == 32 && pointerBits == 64) {
        return OffsetArithmetic::LoweringExtends;
    }
    return std::nullopt;
}

/** True when v is neither undef nor poison as a whole and is provably not
    undef: an SSA value computed from undef may be undef, and each use of
    undef may observe a different value. */
bool lIsNotUndef(llvm::Value *v) { return !llvm::isa<llvm::UndefValue>(v) && llvm::isGuaranteedNotToBeUndef(v); }

} // namespace

std::optional<AddressCandidate> QueryAddressCandidate(const MemoryAccess &access) {
    if (access.form() != MemoryOpForm::BaseOffsets && access.form() != MemoryOpForm::Factored) {
        return std::nullopt;
    }
    llvm::ConstantInt *scale = llvm::dyn_cast<llvm::ConstantInt>(access.scale());
    if (scale == nullptr) {
        return std::nullopt;
    }
    std::optional<OffsetArithmetic> arithmetic = lOffsetArithmetic(access);
    if (!arithmetic) {
        return std::nullopt;
    }
    AddressCandidate candidate;
    candidate.base = access.base();
    candidate.index = access.offsets();
    candidate.indexBits = access.offsetBits();
    candidate.scale = scale->getValue();
    candidate.displacement = access.delta();
    candidate.arithmetic = *arithmetic;
    return candidate;
}

std::optional<NarrowIndex> QueryNarrowIndex(const MemoryAccess &access) {
    if (access.form() != MemoryOpForm::BaseOffsets && access.form() != MemoryOpForm::Factored) {
        return std::nullopt;
    }
    NarrowIndex narrow;
    llvm::Value *index = access.offsets();
    if (access.offsetBits() == 32) {
        narrow.index = index;
    } else if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(index)) {
        // sext(x) and zext nneg (x) both equal sext(x); the operand must be
        // i32 (a cast keeps the lane count).
        bool signExtends = cast->getOpcode() == llvm::Instruction::SExt ||
                           (cast->getOpcode() == llvm::Instruction::ZExt && cast->hasNonNeg());
        if (!signExtends || !cast->getOperand(0)->getType()->getScalarType()->isIntegerTy(32)) {
            return std::nullopt;
        }
        narrow.index = cast->getOperand(0);
    } else {
        llvm::SmallVector<llvm::APInt, ISPC_MAX_NVEC> lanes;
        if (!lConstantLanes(index, lanes) || !lNarrowLanes(lanes, narrow.indexLanes)) {
            return std::nullopt;
        }
    }
    if (access.delta() != nullptr) {
        llvm::SmallVector<llvm::APInt, ISPC_MAX_NVEC> lanes;
        if (!lConstantLanes(access.delta(), lanes) || !lNarrowLanes(lanes, narrow.displacementLanes)) {
            return std::nullopt;
        }
    }
    return narrow;
}

LanePattern QueryLanePattern(const MemoryAccess &access) {
    LanePattern pattern;
    pattern.mask = access.maskStatus();
    llvm::Value *v = nullptr;
    if (access.form() == MemoryOpForm::General) {
        v = access.addresses();
        pattern.operand = "addresses";
    } else if (access.form() == MemoryOpForm::BaseOffsets || access.form() == MemoryOpForm::Factored) {
        v = access.offsets();
        pattern.operand = "offsets";
    } else {
        return pattern;
    }
    llvm::SmallVector<llvm::APInt, ISPC_MAX_NVEC> lanes;
    if (lConstantLanes(v, lanes)) {
        pattern.coversAllLanes = true;
        pattern.lane0Defined = true;
        pattern.shape = LanePattern::Shape::AllEqual;
        if (lanes.size() > 1) {
            llvm::APInt stride = lanes[1] - lanes[0];
            for (unsigned i = 1; i < lanes.size(); ++i) {
                if (lanes[i] - lanes[i - 1] != stride) {
                    pattern.shape = LanePattern::Shape::Unknown;
                    return pattern;
                }
            }
            if (!stride.isZero()) {
                pattern.shape = LanePattern::Shape::Linear;
                pattern.stride = stride;
            }
        }
        return pattern;
    }
    // A shuffle that broadcasts one lane of its first operand to every lane:
    // all lanes are the same value, which may be poison, so lane 0 is not
    // known to be defined. A poison mask lane would make that lane poison on
    // its own, so every mask lane must select the same lane.
    // An undef source lane would let every lane differ.
    if (llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(v)) {
        llvm::ArrayRef<int> mask = shuffle->getShuffleMask();
        if (mask.empty() || mask[0] < 0 || !llvm::all_of(mask, [&](int m) { return m == mask[0]; })) {
            return pattern;
        }
        llvm::Value *source = shuffle->getOperand(0);
        unsigned lane = mask[0];
        unsigned sourceLanes = llvm::cast<llvm::FixedVectorType>(source->getType())->getNumElements();
        if (lane >= sourceLanes) {
            source = shuffle->getOperand(1);
            lane -= sourceLanes;
        }
        // The source lane must be a value that is provably not undef, inserted
        // there or a constant lane: LLVM folds a broadcast of an undef lane,
        // or of a value computed from undef, to an undef vector, whose lanes
        // need not agree. Poison is rejected as well.
        while (llvm::InsertElementInst *insert = llvm::dyn_cast<llvm::InsertElementInst>(source)) {
            llvm::ConstantInt *index = llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2));
            if (index == nullptr) {
                return pattern;
            }
            if (index->getValue() == lane) {
                if (!lIsNotUndef(insert->getOperand(1))) {
                    return pattern;
                }
                pattern.shape = LanePattern::Shape::AllEqual;
                pattern.coversAllLanes = true;
                return pattern;
            }
            source = insert->getOperand(0);
        }
        llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(source);
        llvm::Constant *element = c == nullptr ? nullptr : c->getAggregateElement(lane);
        if (element == nullptr || !lIsNotUndef(element)) {
            return pattern;
        }
        pattern.shape = LanePattern::Shape::AllEqual;
        pattern.coversAllLanes = true;
    }
    return pattern;
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
    if (form() == MemoryOpForm::General) {
        if (std::optional<BaseOffsetsRecipe> recipe = RecoverBaseOffsets(*this)) {
            lPrintRecipe(os, *recipe);
        } else {
            os << "  common base: unknown\n";
        }
    }
    if (std::optional<AddressCandidate> candidate = QueryAddressCandidate(*this)) {
        os << "  candidate: base ";
        candidate->base->printAsOperand(os, /*PrintType=*/true);
        os << " + index ";
        candidate->index->printAsOperand(os, /*PrintType=*/true);
        os << " * " << candidate->scale;
        if (candidate->displacement != nullptr) {
            os << " + displacement ";
            candidate->displacement->printAsOperand(os, /*PrintType=*/true);
        }
        if (candidate->arithmetic == OffsetArithmetic::AddressWidth) {
            os << " (address width)\n";
        } else {
            os << " (lowering extends i" << candidate->indexBits << ")\n";
        }
        os << "  narrow index: ";
        if (std::optional<NarrowIndex> narrow = QueryNarrowIndex(*this)) {
            if (narrow->index != nullptr) {
                narrow->index->printAsOperand(os, /*PrintType=*/true);
            } else {
                lPrintLanes(os, narrow->indexLanes);
            }
            if (!narrow->displacementLanes.empty()) {
                os << ", displacement ";
                lPrintLanes(os, narrow->displacementLanes);
            }
            os << "\n";
        } else {
            os << "unknown\n";
        }
    }
    LanePattern pattern = QueryLanePattern(*this);
    if (pattern.operand[0] != '\0') {
        os << "  lane pattern of " << pattern.operand << ": ";
        switch (pattern.shape) {
        case LanePattern::Shape::Unknown:
            os << "unknown";
            break;
        case LanePattern::Shape::AllEqual:
            os << "all equal";
            break;
        case LanePattern::Shape::Linear:
            os << "linear, stride " << pattern.stride;
            break;
        }
        if (pattern.shape != LanePattern::Shape::Unknown) {
            os << (pattern.coversAllLanes ? ", all lanes" : ", active lanes");
        }
        os << "; lane 0 " << (pattern.lane0Defined ? "defined" : "not known defined") << "\n";
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
