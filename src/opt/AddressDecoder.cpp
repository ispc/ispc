/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "AddressDecoder.h"

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/Operator.h>

namespace ispc {

namespace {

/** How many values one decode visits at most. Real addresses are a few
    dozen values; beyond the budget the address is unknown. */
constexpr int kDecodeBudget = 256;

/** One decode of a vector of pointers. */
struct Decoder {
    const llvm::DataLayout &DL;
    unsigned lanes;
    unsigned indexBits;
    int budget = kDecodeBudget;
    AddressDecode result;

    Decoder(const llvm::DataLayout &DL, unsigned lanes, unsigned indexBits)
        : DL(DL), lanes(lanes), indexBits(indexBits) {
        result.indexBits = indexBits;
        result.laneOffsets.assign(lanes, llvm::APInt(indexBits, 0));
    }

    bool visit() { return --budget >= 0; }

    /** Adds c * coefficient to every lane. */
    void addConstant(const llvm::APInt &c, const llvm::APInt &coefficient) {
        llvm::APInt bytes = c.sextOrTrunc(indexBits) * coefficient;
        for (llvm::APInt &lane : result.laneOffsets) {
            lane += bytes;
        }
    }

    /** The lanes of a constant index, sign extended or truncated to the index
        width like a GEP does; false for undef or poison lanes. */
    bool constantLanes(llvm::Constant *c, llvm::SmallVectorImpl<llvm::APInt> &out) {
        if (c->containsUndefOrPoisonElement()) {
            return false;
        }
        if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(c)) {
            out.assign(lanes, ci->getValue().sextOrTrunc(indexBits));
            return true;
        }
        llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(c->getType());
        if (vt == nullptr || vt->getNumElements() != lanes || !vt->getElementType()->isIntegerTy()) {
            return false;
        }
        for (unsigned i = 0; i < lanes; ++i) {
            llvm::ConstantInt *ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(c->getAggregateElement(i));
            if (ci == nullptr) {
                return false;
            }
            out.push_back(ci->getValue().sextOrTrunc(indexBits));
        }
        return true;
    }

    /** Adds the lanes of a constant index times coefficient. */
    bool addConstantIndex(llvm::Constant *c, const llvm::APInt &coefficient) {
        llvm::SmallVector<llvm::APInt, ISPC_MAX_NVEC> values;
        if (!constantLanes(c, values)) {
            return false;
        }
        for (unsigned i = 0; i < lanes; ++i) {
            result.laneOffsets[i] += values[i] * coefficient;
        }
        return true;
    }

    /** Records one GEP index v, at any width, with the given coefficient.
        Constants go to the lane offsets. At the index width, an add (or a
        disjoint or) is split into its operands and a mul or shl by a constant
        goes into the coefficient: both are exact, as the GEP computes at the
        index width too. Nothing is looked through an extension. */
    bool addIndex(llvm::Value *v, llvm::APInt coefficient) {
        if (!visit()) {
            return false;
        }
        if (coefficient.isZero()) {
            return true;
        }
        if (llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(v)) {
            if (c->containsUndefOrPoisonElement()) {
                return false;
            }
            if (llvm::Constant *splat = c->getType()->isVectorTy() ? c->getSplatValue() : nullptr) {
                c = splat;
            }
            if (addConstantIndex(c, coefficient)) {
                return true;
            }
            // A constant expression index; kept as a term below.
        }
        // A splat of a scalar is a uniform index.
        bool uniform = !v->getType()->isVectorTy();
        if (!uniform) {
            if (llvm::Value *splat = llvm::getSplatValue(v)) {
                v = splat;
                uniform = true;
            }
        }
        unsigned bits = v->getType()->getScalarSizeInBits();
        if (bits == indexBits) {
            if (llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v)) {
                llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
                if (bop->getOpcode() == llvm::Instruction::Add || IsOrEquivalentToAdd(bop)) {
                    return addIndex(op0, coefficient) && addIndex(op1, coefficient);
                }
                if (bop->getOpcode() == llvm::Instruction::Sub) {
                    return addIndex(op0, coefficient) && addIndex(op1, -coefficient);
                }
                if (bop->getOpcode() == llvm::Instruction::Mul || bop->getOpcode() == llvm::Instruction::Shl) {
                    llvm::Value *other = op0;
                    llvm::ConstantInt *ci = lConstantSplat(op1);
                    if (ci == nullptr && bop->getOpcode() == llvm::Instruction::Mul) {
                        ci = lConstantSplat(op0);
                        other = op1;
                    }
                    if (ci != nullptr) {
                        llvm::APInt factor = ci->getValue().sextOrTrunc(indexBits);
                        if (bop->getOpcode() == llvm::Instruction::Shl) {
                            if (factor.uge(indexBits)) {
                                return false; // poison
                            }
                            factor = llvm::APInt::getOneBitSet(indexBits, factor.getZExtValue());
                        }
                        return addIndex(other, coefficient * factor);
                    }
                }
            }
        }
        AddressTerm term;
        term.index = v;
        term.uniform = uniform;
        term.coefficient = coefficient;
        if (bits < indexBits) {
            // The GEP sign extends a narrower index.
            term.extension = AddressTerm::Extension::Sign;
            term.narrow = v;
        } else if (bits > indexBits) {
            // The GEP truncates a wider index: not a form the decoder knows.
            return false;
        } else if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(v)) {
            if (cast->getOpcode() == llvm::Instruction::SExt) {
                term.extension = AddressTerm::Extension::Sign;
                term.narrow = cast->getOperand(0);
            } else if (cast->getOpcode() == llvm::Instruction::ZExt) {
                term.extension = cast->hasNonNeg() ? AddressTerm::Extension::ZeroNonNeg : AddressTerm::Extension::Zero;
                term.narrow = cast->getOperand(0);
            }
        }
        result.terms.push_back(term);
        return true;
    }

    static llvm::ConstantInt *lConstantSplat(llvm::Value *v) {
        llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(v);
        if (c == nullptr) {
            return nullptr;
        }
        if (c->getType()->isVectorTy()) {
            c = c->getSplatValue();
        }
        return llvm::dyn_cast_or_null<llvm::ConstantInt>(c);
    }

    /** Decodes the indices of a GEP, after its base. */
    bool gepIndices(llvm::GEPOperator *gep) {
        for (llvm::gep_type_iterator it = llvm::gep_type_begin(gep), end = llvm::gep_type_end(gep); it != end; ++it) {
            llvm::Value *index = it.getOperand();
            if (llvm::StructType *st = it.getStructTypeOrNull()) {
                // A struct index is a constant, or a splat of one in a vector GEP.
                llvm::ConstantInt *field = lConstantSplat(index);
                if (field == nullptr) {
                    return false;
                }
                uint64_t offset = DL.getStructLayout(st)->getElementOffset(field->getZExtValue()).getFixedValue();
                addConstant(llvm::APInt(indexBits, offset), llvm::APInt(indexBits, 1));
                continue;
            }
            llvm::TypeSize stride = it.getSequentialElementStride(DL);
            if (stride.isScalable()) {
                return false;
            }
            if (!addIndex(index, llvm::APInt(indexBits, stride.getFixedValue()))) {
                return false;
            }
        }
        return true;
    }

    /** Sets a uniform base: the scalar pointer p. */
    void uniformBase(llvm::Value *p) {
        result.base = AddressBase::Uniform;
        result.origin = AddressOrigin::Scalar;
        result.root = p;
    }

    /** Decodes a vector of pointers. */
    bool decode(llvm::Value *v) {
        if (!visit()) {
            return false;
        }
        // A scalar base, as the pointer operand of a vector GEP.
        if (!v->getType()->isVectorTy()) {
            uniformBase(v);
            return true;
        }
        if (llvm::Value *splat = llvm::getSplatValue(v)) {
            if (llvm::isa<llvm::UndefValue>(splat)) {
                return false;
            }
            uniformBase(splat);
            return true;
        }
        if (llvm::GEPOperator *gep = llvm::dyn_cast<llvm::GEPOperator>(v)) {
            result.viaGEP = true;
            return decode(gep->getPointerOperand()) && gepIndices(gep);
        }
        if (llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(v)) {
            return constantVector(c);
        }
        if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(v)) {
            switch (cast->getOpcode()) {
            case llvm::Instruction::AddrSpaceCast: {
                // Only between address spaces of the same widths: the GEPs
                // under the cast compute at their own index width.
                unsigned from = cast->getOperand(0)->getType()->getPointerAddressSpace();
                unsigned to = cast->getType()->getPointerAddressSpace();
                if (DL.getIndexSizeInBits(from) != DL.getIndexSizeInBits(to) ||
                    DL.getPointerSizeInBits(from) != DL.getPointerSizeInBits(to)) {
                    break;
                }
                return decode(cast->getOperand(0));
            }
            case llvm::Instruction::BitCast:
                return decode(cast->getOperand(0));
            case llvm::Instruction::IntToPtr:
                result.base = AddressBase::Integer;
                result.origin = AddressOrigin::IntToPtr;
                result.root = v;
                return true;
            default:
                break;
            }
        }
        result.base = AddressBase::PerLane;
        result.root = v;
        if (llvm::isa<llvm::LoadInst>(v)) {
            result.origin = AddressOrigin::Load;
        } else if (llvm::isa<llvm::Argument>(v)) {
            result.origin = AddressOrigin::Argument;
        } else if (llvm::isa<llvm::PHINode>(v) || llvm::isa<llvm::SelectInst>(v)) {
            result.origin = AddressOrigin::PhiSelect;
        } else if (llvm::isa<llvm::CallBase>(v)) {
            result.origin = AddressOrigin::Call;
        } else {
            result.origin = AddressOrigin::Other;
        }
        return true;
    }

    /** Whether every address space cast that the strip of constant offsets
        looks through in v is between spaces of the same widths. As for an
        address space cast instruction: the strip sign extends an offset
        computed in a space of other widths, but the cast may zero extend
        the pointer (null - 4 in a 32-bit space is 0xFFFFFFFC). */
    bool castsKeepWidths(const llvm::Value *v) const {
        while (true) {
            if (const llvm::GEPOperator *gep = llvm::dyn_cast<llvm::GEPOperator>(v)) {
                v = gep->getPointerOperand();
                continue;
            }
            if (const llvm::Operator *op = llvm::dyn_cast<llvm::Operator>(v)) {
                if (op->getOpcode() == llvm::Instruction::BitCast) {
                    v = op->getOperand(0);
                    continue;
                }
                if (op->getOpcode() == llvm::Instruction::AddrSpaceCast) {
                    unsigned from = op->getOperand(0)->getType()->getPointerAddressSpace();
                    unsigned to = op->getType()->getPointerAddressSpace();
                    if (DL.getIndexSizeInBits(from) != DL.getIndexSizeInBits(to) ||
                        DL.getPointerSizeInBits(from) != DL.getPointerSizeInBits(to)) {
                        return false;
                    }
                    v = op->getOperand(0);
                    continue;
                }
            }
            if (const llvm::GlobalAlias *alias = llvm::dyn_cast<llvm::GlobalAlias>(v)) {
                if (!alias->isInterposable()) {
                    v = alias->getAliasee();
                    continue;
                }
            }
            return true;
        }
    }

    /** A constant vector whose lanes are pointers into one global, with
        constant offsets. */
    bool constantVector(llvm::Constant *c) {
        if (c->containsUndefOrPoisonElement()) {
            return false;
        }
        llvm::Value *base = nullptr;
        for (unsigned i = 0; i < lanes; ++i) {
            llvm::Constant *lane = c->getAggregateElement(i);
            if (lane == nullptr) {
                return false;
            }
            llvm::APInt offset(indexBits, 0);
            llvm::Value *laneBase = lane->stripAndAccumulateConstantOffsets(DL, offset, /*AllowNonInbounds=*/true);
            if (!castsKeepWidths(lane)) {
                return false;
            }
            if (base == nullptr) {
                base = laneBase;
            } else if (laneBase != base) {
                // Different lanes point into different objects.
                result.base = AddressBase::PerLane;
                result.origin = AddressOrigin::Other;
                result.root = c;
                result.laneOffsets.assign(lanes, llvm::APInt(indexBits, 0));
                return true;
            }
            result.laneOffsets[i] += offset.sextOrTrunc(indexBits);
        }
        uniformBase(base);
        return true;
    }

    /** The object facts of a uniform base. */
    void object() {
        if (result.base != AddressBase::Uniform) {
            return;
        }
        result.oneObject = true;
        // The size is reported only with the offset of the root in the
        // object: the lanes are offsets from the root.
        unsigned bits = DL.getIndexSizeInBits(result.root->getType()->getPointerAddressSpace());
        llvm::APInt offset(bits, 0);
        const llvm::Value *object =
            result.root->stripAndAccumulateConstantOffsets(DL, offset, /*AllowNonInbounds=*/true);
        uint64_t bytes = 0;
        if (const llvm::AllocaInst *alloca = llvm::dyn_cast<llvm::AllocaInst>(object)) {
            if (!alloca->getAllocatedType()->isScalableTy()) {
                if (std::optional<llvm::TypeSize> size = alloca->getAllocationSize(DL)) {
                    bytes = size->getFixedValue();
                }
            }
        } else if (const llvm::GlobalVariable *gv = llvm::dyn_cast<llvm::GlobalVariable>(object)) {
            if (gv->getValueType()->isSized() && !gv->getValueType()->isScalableTy() && !gv->isInterposable()) {
                bytes = DL.getTypeAllocSize(gv->getValueType()).getFixedValue();
            }
        }
        if (bytes != 0) {
            result.objectBytes = bytes;
            result.rootOffset = offset.getSExtValue();
        }
    }
};

const char *lBaseName(AddressBase base) {
    switch (base) {
    case AddressBase::Uniform:
        return "uniform";
    case AddressBase::PerLane:
        return "per-lane";
    case AddressBase::Integer:
        return "integer";
    }
    return "?";
}

const char *lOriginName(AddressOrigin origin) {
    switch (origin) {
    case AddressOrigin::Scalar:
        return "scalar";
    case AddressOrigin::IntToPtr:
        return "inttoptr";
    case AddressOrigin::Load:
        return "load";
    case AddressOrigin::Argument:
        return "argument";
    case AddressOrigin::PhiSelect:
        return "phi/select";
    case AddressOrigin::Call:
        return "call";
    case AddressOrigin::Other:
        return "other";
    }
    return "?";
}

const char *lExtensionName(AddressTerm::Extension extension) {
    switch (extension) {
    case AddressTerm::Extension::None:
        return "";
    case AddressTerm::Extension::Sign:
        return "sext ";
    case AddressTerm::Extension::Zero:
        return "zext ";
    case AddressTerm::Extension::ZeroNonNeg:
        return "zext nneg ";
    }
    return "?";
}

} // namespace

bool AddressDecode::onlyUniformTerms() const {
    return llvm::all_of(terms, [](const AddressTerm &term) { return term.uniform; });
}

bool AddressDecode::laneOffsetsEqual() const {
    return llvm::all_of(laneOffsets, [&](const llvm::APInt &lane) { return lane == laneOffsets[0]; });
}

void AddressDecode::print(llvm::raw_ostream &os) const {
    os << "  address: " << lBaseName(base) << " base ";
    root->printAsOperand(os, /*PrintType=*/true);
    os << " (" << lOriginName(origin) << (viaGEP ? ", gep" : "") << ")";
    if (oneObject) {
        os << ", one object";
        if (objectBytes != 0) {
            os << " of " << objectBytes << " bytes";
            if (rootOffset != 0) {
                os << " at offset " << rootOffset;
            }
        }
    }
    os << "\n";
    for (const AddressTerm &term : terms) {
        os << "    term: " << (term.uniform ? "uniform " : "") << lExtensionName(term.extension);
        (term.narrow != nullptr ? term.narrow : term.index)->printAsOperand(os, /*PrintType=*/true);
        os << " * " << term.coefficient.getSExtValue() << "\n";
    }
    os << "    lanes:";
    for (const llvm::APInt &lane : laneOffsets) {
        os << " " << lane.getSExtValue();
    }
    os << " (i" << indexBits << ")\n";
}

std::optional<AddressDecode> DecodeVaryingAddress(llvm::Value *ptrs, const llvm::DataLayout &DL) {
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(ptrs->getType());
    if (vt == nullptr || !vt->getElementType()->isPointerTy()) {
        return std::nullopt;
    }
    unsigned indexBits = DL.getIndexSizeInBits(vt->getElementType()->getPointerAddressSpace());
    Decoder decoder(DL, vt->getNumElements(), indexBits);
    if (!decoder.decode(ptrs)) {
        return std::nullopt;
    }
    decoder.object();
    return std::move(decoder.result);
}

llvm::Value *GetPointerAddresses(llvm::Value *addresses, const llvm::DataLayout &DL) {
    llvm::Type *t = addresses->getType();
    if (t->isVectorTy() && t->getScalarType()->isPointerTy()) {
        return addresses;
    }
    if (llvm::PtrToIntOperator *p2i = llvm::dyn_cast<llvm::PtrToIntOperator>(addresses)) {
        llvm::Value *ptrs = p2i->getPointerOperand();
        // A ptrtoint that truncates the pointers has other values.
        if (ptrs->getType()->isVectorTy() &&
            t->getScalarSizeInBits() == DL.getPointerSizeInBits(ptrs->getType()->getPointerAddressSpace())) {
            return ptrs;
        }
    }
    return nullptr;
}

} // namespace ispc
