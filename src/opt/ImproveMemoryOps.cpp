/*
  Copyright (c) 2022-2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ImproveMemoryOps.h"
#include "MemoryAccess.h"

#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/VectorUtils.h>

namespace ispc {

/** Returns true if the parts that lExtractConstantOffset() splits the vector
    expression in vec into can't wrap as unsigned values, i.e. each part and
    each partial sum is at most vec. This holds if every add, shl and mul it
    looks through is nuw. Only then can a zext be applied to the parts instead
    of to their sum.
 */
static bool lSplitIsNUW(llvm::Value *vec) {
    if (llvm::isa<llvm::Constant>(vec)) {
        return true;
    }
    if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(vec)) {
        return cast->getOpcode() == llvm::Instruction::ZExt && lSplitIsNUW(cast->getOperand(0));
    }
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bop == nullptr) {
        return true;
    }
    switch (bop->getOpcode()) {
    case llvm::Instruction::Add:
    case llvm::Instruction::Shl:
    case llvm::Instruction::Mul:
        return bop->hasNoUnsignedWrap() && lSplitIsNUW(bop->getOperand(0)) && lSplitIsNUW(bop->getOperand(1));
    default:
        // A disjoint or never wraps; anything else is not split.
        return !IsOrEquivalentToAdd(bop) || (lSplitIsNUW(bop->getOperand(0)) && lSplitIsNUW(bop->getOperand(1)));
    }
}

/** Returns true if every add, shl and mul that lExtractConstantOffset()
    looks through in vec is nsw (a disjoint or can't wrap). Then sign
    extending the parts gives the sign extension of their sum. A zext inside
    vec is split exactly or not at all, and a sext only under this rule. A
    trunc is split modulo 2^N: trunc(x + INT32_MIN) with an i64 x = INT32_MIN
    is 0, but sign extending its parts gives -2^32.
 */
static bool lSplitIsNSW(llvm::Value *vec) {
    if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(vec)) {
        switch (cast->getOpcode()) {
        case llvm::Instruction::SExt:
            return lSplitIsNSW(cast->getOperand(0));
        case llvm::Instruction::ZExt:
            return true;
        default:
            return false;
        }
    }
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bop == nullptr) {
        return true;
    }
    switch (bop->getOpcode()) {
    case llvm::Instruction::Add:
    case llvm::Instruction::Shl:
    case llvm::Instruction::Mul:
        return bop->hasNoSignedWrap() && lSplitIsNSW(bop->getOperand(0)) && lSplitIsNSW(bop->getOperand(1));
    default:
        return !IsOrEquivalentToAdd(bop) || (lSplitIsNSW(bop->getOperand(0)) && lSplitIsNSW(bop->getOperand(1)));
    }
}

/** Returns false if lExtractConstantOffset() can't split a constant part
    off vec. May return true even if it can't. */
static bool lHasConstantPart(llvm::Value *vec) {
    if (llvm::isa<llvm::Constant>(vec)) {
        return true;
    }
    if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(vec)) {
        return lHasConstantPart(cast->getOperand(0));
    }
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bop == nullptr) {
        return false;
    }
    llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
    if (bop->getOpcode() == llvm::Instruction::Add || IsOrEquivalentToAdd(bop)) {
        return lHasConstantPart(op0) || lHasConstantPart(op1);
    }
    if (bop->getOpcode() == llvm::Instruction::Shl) {
        return lHasConstantPart(op0);
    }
    if (bop->getOpcode() == llvm::Instruction::Mul) {
        return lHasConstantPart(op0) && lHasConstantPart(op1);
    }
    return false;
}

/** Folds v to a constant if it is a tree of side-effect-free instructions
    over constants, as lExtractConstantOffset() builds for constant parts. */
static llvm::Constant *lFoldToConstant(llvm::Value *v, const llvm::DataLayout &dl, int depth = 0) {
    if (llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(v)) {
        return c;
    }
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    if (inst == nullptr || depth > 16 || llvm::isa<llvm::PHINode>(inst) || llvm::isa<llvm::CallBase>(inst) ||
        inst->mayReadOrWriteMemory()) {
        return nullptr;
    }
    llvm::SmallVector<llvm::Constant *, 4> ops;
    for (llvm::Value *op : inst->operands()) {
        llvm::Constant *c = lFoldToConstant(op, dl, depth + 1);
        if (c == nullptr) {
            return nullptr;
        }
        ops.push_back(c);
    }
    return llvm::ConstantFoldInstOperands(inst, ops, dl);
}

/** Computes in *out the exact value of lane `lane` of the constant part c
    that lExtractConstantOffset() built: the leaves are sign extended and the
    adds, shls and muls it regroups are done in i64, with overflow checks.
    Only valid if the expression c was split from passes lSplitIsNSW(): then
    each of those operations, and each sext it looks through, is nsw in the
    original. The regrouped narrow constant can wrap where the original sum
    doesn't: (x + 0x40000000) + 0x40000000 regroups 0x40000000 + 0x40000000,
    which is INT32_MIN at i32 width. Returns false if c doesn't fold or
    overflows i64. */
static bool lEvalExact(llvm::Value *c, unsigned lane, llvm::APInt *out, const llvm::DataLayout &dl, int depth = 0) {
    if (depth > 16) {
        return false;
    }
    if (llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(c)) {
        llvm::Instruction::BinaryOps opcode = bop->getOpcode();
        if (opcode == llvm::Instruction::Add || opcode == llvm::Instruction::Shl || opcode == llvm::Instruction::Mul ||
            IsOrEquivalentToAdd(bop)) {
            llvm::APInt op0;
            if (!lEvalExact(bop->getOperand(0), lane, &op0, dl, depth + 1)) {
                return false;
            }
            bool overflow = false;
            if (opcode == llvm::Instruction::Shl) {
                llvm::Constant *amount = lFoldToConstant(bop->getOperand(1), dl);
                llvm::ConstantInt *ci =
                    amount ? llvm::dyn_cast_or_null<llvm::ConstantInt>(amount->getAggregateElement(lane)) : nullptr;
                if (ci == nullptr || ci->getValue().uge(64)) {
                    return false;
                }
                *out = op0.sshl_ov(ci->getValue().zextOrTrunc(64), overflow);
                return !overflow;
            }
            llvm::APInt op1;
            if (!lEvalExact(bop->getOperand(1), lane, &op1, dl, depth + 1)) {
                return false;
            }
            *out = opcode == llvm::Instruction::Mul ? op0.smul_ov(op1, overflow) : op0.sadd_ov(op1, overflow);
            return !overflow;
        }
    }
    if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(c)) {
        if (cast->getOpcode() == llvm::Instruction::SExt) {
            return lEvalExact(cast->getOperand(0), lane, out, dl, depth + 1);
        }
        if (cast->getOpcode() == llvm::Instruction::Trunc) {
            return false;
        }
    }
    // Anything else isn't regrouped: fold it at its own width and extend it.
    // (A zext result has a clear sign bit, so sign extending it is exact.)
    llvm::Constant *folded = lFoldToConstant(c, dl);
    llvm::ConstantInt *ci =
        folded ? llvm::dyn_cast_or_null<llvm::ConstantInt>(folded->getAggregateElement(lane)) : nullptr;
    if (ci == nullptr || ci->getBitWidth() > 64) {
        return false;
    }
    *out = ci->getValue().sext(64);
    return true;
}

/** Returns true if every lane of the constant part c, evaluated exactly
    (lEvalExact()), fits in a signed integer of the given width. */
static bool lExactFitsSigned(llvm::Value *c, unsigned width, const llvm::DataLayout &dl) {
    unsigned n = llvm::cast<llvm::FixedVectorType>(c->getType())->getNumElements();
    for (unsigned i = 0; i < n; ++i) {
        llvm::APInt value;
        if (!lEvalExact(c, i, &value, dl) || !value.isSignedIntN(width)) {
            return false;
        }
    }
    return true;
}

/** Returns true if every lane of the integer vector constant c fits in a
    signed i32. */
static bool lConstantFits32(llvm::Constant *c) {
    unsigned n = llvm::cast<llvm::FixedVectorType>(c->getType())->getNumElements();
    for (unsigned i = 0; i < n; ++i) {
        llvm::ConstantInt *ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(c->getAggregateElement(i));
        if (ci == nullptr || !ci->getValue().isSignedIntN(32)) {
            return false;
        }
    }
    return true;
}

/** Given a vector expression in vec, separate it into a compile-time
    constant component and a variable component, returning the two parts in
    *constOffset and *variableOffset.  (It should be the case that the sum
    of these two is exactly equal to the original vector.)

    This routine only handles some (important) patterns; in some cases it
    will fail and return components that are actually compile-time
    constants in *variableOffset.

    Finally, if there aren't any constant (or, respectivaly, variable)
    components, the corresponding return value may be set to nullptr.
 */
static void lExtractConstantOffset(llvm::Value *vec, llvm::Value **constOffset, llvm::Value **variableOffset,
                                   llvm::Instruction *insertBefore) {
    if (llvm::isa<llvm::ConstantVector>(vec) || llvm::isa<llvm::ConstantDataVector>(vec) ||
        llvm::isa<llvm::ConstantAggregateZero>(vec)) {
        *constOffset = vec;
        *variableOffset = nullptr;
        return;
    }

    llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(vec);
    if (cast != nullptr) {
        llvm::Instruction::CastOps opcode = cast->getOpcode();
        if (opcode == llvm::Instruction::ZExt) {
            if (cast->hasNonNeg() && lSplitIsNSW(cast->getOperand(0))) {
                // InstCombine turns a sext of a value that is known to be
                // non-negative into a zext nneg. nneg bounds only the sum and
                // one of the parts may be negative, so the parts are sign
                // extended. That is exact only if the sum doesn't wrap as a
                // signed value: x + INT32_MIN wraps to zero for x = INT32_MIN.
                opcode = llvm::Instruction::SExt;
            } else if (!lSplitIsNUW(cast->getOperand(0))) {
                // zext(c + v) is zext(c) + zext(v) only if c + v doesn't wrap.
                // For example, p[i - 1] with an unsigned i is zext(i + 0xFFFFFFFF).
                *constOffset = nullptr;
                *variableOffset = vec;
                return;
            }
        }

        // Check the cast target.
        llvm::Value *co = nullptr, *vo = nullptr;
        lExtractConstantOffset(cast->getOperand(0), &co, &vo, insertBefore);

        // nsw holds for the original grouping, not for the regrouped
        // constant: sign extending it is exact only if its exact value fits.
        // (Without nsw, a sext is split modulo 2^N, as before.)
        if (opcode == llvm::Instruction::SExt && co != nullptr && lSplitIsNSW(cast->getOperand(0))) {
            const llvm::DataLayout &dl = insertBefore->getModule()->getDataLayout();
            if (!lExactFitsSigned(co, co->getType()->getScalarSizeInBits(), dl)) {
                *constOffset = nullptr;
                *variableOffset = vec;
                return;
            }
        }

        // make new cast instructions for the two parts
        if (co == nullptr) {
            *constOffset = nullptr;
        } else {
            *constOffset = llvm::CastInst::Create(opcode, co, cast->getType(), llvm::Twine(co->getName()) + "_cast",
                                                  ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
        }
        if (vo == nullptr) {
            *variableOffset = nullptr;
        } else {
            *variableOffset = llvm::CastInst::Create(opcode, vo, cast->getType(), llvm::Twine(vo->getName()) + "_cast",
                                                     ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
        }
        return;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bop != nullptr) {
        llvm::Value *op0 = bop->getOperand(0);
        llvm::Value *op1 = bop->getOperand(1);
        llvm::Value *c0 = nullptr, *v0 = nullptr, *c1 = nullptr, *v1 = nullptr;

        if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            if (c0 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(c0)) {
                *constOffset = c1;
            } else if (c1 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(c1)) {
                *constOffset = c0;
            } else {
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, c0, c1, ((llvm::Twine("add_") + c0->getName()) + "_") + c1->getName(),
                    ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            }

            if (v0 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(v0)) {
                *variableOffset = v1;
            } else if (v1 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(v1)) {
                *variableOffset = v0;
            } else {
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, v0, v1, ((llvm::Twine("add_") + v0->getName()) + "_") + v1->getName(),
                    ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            }
            return;
        } else if (bop->getOpcode() == llvm::Instruction::Shl) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            // Given the product of constant and variable terms, we have:
            // (c0 + v0) * (2^(c1 + v1))  = c0 * 2^c1 * 2^v1 + v0 * 2^c1 * 2^v1
            // We can optimize only if v1 == nullptr.
            if ((v1 != nullptr) || (c0 == nullptr) || (c1 == nullptr)) {
                *constOffset = nullptr;
                *variableOffset = vec;
            } else if (v0 == nullptr) {
                *constOffset = vec;
                *variableOffset = nullptr;
            } else {
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Shl, c0, c1, ((llvm::Twine("shl_") + c0->getName()) + "_") + c1->getName(),
                    ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
                // Copy flags (nsw) from the original instruction only if
                // --wrap-signed-int is disabled (this is the source of the nsw
                // flags we want to transfer to the new instruction)
                if (!g->wrapSignedInt) {
                    llvm::BinaryOperator *shl = llvm::BinaryOperator::CreateWithCopiedFlags(
                        llvm::Instruction::Shl, v0, c1, bop,
                        ((llvm::Twine("shl_") + v0->getName()) + "_") + c1->getName(),
                        ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
                    // v0 may be larger than c0 + v0 if that sum wraps, and
                    // then v0 << c1 may wrap where (c0 + v0) << c1 doesn't.
                    if (!lSplitIsNUW(op0)) {
                        shl->setHasNoUnsignedWrap(false);
                    }
                    *variableOffset = shl;
                } else {
                    *variableOffset = llvm::BinaryOperator::Create(
                        llvm::Instruction::Shl, v0, c1, ((llvm::Twine("shl_") + v0->getName()) + "_") + c1->getName(),
                        ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
                }
            }
            return;
        } else if (bop->getOpcode() == llvm::Instruction::Mul) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            // Given the product of constant and variable terms, we have:
            // (c0 + v0) * (c1 + v1) == (c0 c1) + (v0 c1 + c0 v1 + v0 v1)
            // Note that the first term is a constant and the last three are
            // variable.
            if (c0 != nullptr && c1 != nullptr) {
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Mul, c0, c1, ((llvm::Twine("mul_") + c0->getName()) + "_") + c1->getName(),
                    ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            } else {
                *constOffset = nullptr;
            }

            llvm::Value *va = nullptr, *vb = nullptr, *vc = nullptr;
            if (v0 != nullptr && c1 != nullptr) {
                va = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, c1,
                                                  ((llvm::Twine("mul_") + v0->getName()) + "_") + c1->getName(),
                                                  ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            }
            if (c0 != nullptr && v1 != nullptr) {
                vb = llvm::BinaryOperator::Create(llvm::Instruction::Mul, c0, v1,
                                                  ((llvm::Twine("mul_") + c0->getName()) + "_") + v1->getName(),
                                                  ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            }
            if (v0 != nullptr && v1 != nullptr) {
                vc = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, v1,
                                                  ((llvm::Twine("mul_") + v0->getName()) + "_") + v1->getName(),
                                                  ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            }

            llvm::Value *vab = nullptr;
            if (va != nullptr && vb != nullptr) {
                vab = llvm::BinaryOperator::Create(llvm::Instruction::Add, va, vb,
                                                   ((llvm::Twine("add_") + va->getName()) + "_") + vb->getName(),
                                                   ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            } else if (va != nullptr) {
                vab = va;
            } else {
                vab = vb;
            }

            if (vab != nullptr && vc != nullptr) {
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, vab, vc, ((llvm::Twine("add_") + vab->getName()) + "_") + vc->getName(),
                    ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            } else if (vab != nullptr) {
                *variableOffset = vab;
            } else {
                *variableOffset = vc;
            }

            return;
        }
    }

    // Nothing matched, just return what we have as a variable component
    *constOffset = nullptr;
    *variableOffset = vec;
}

/* Returns true if the given value is a constant vector of integers with
   the same value in all of the elements.  (Returns the splatted value in
   *splat, if so). */
static bool lIsIntegerSplat(llvm::Value *v, int *splat) {
    llvm::ConstantDataVector *cvec = llvm::dyn_cast<llvm::ConstantDataVector>(v);
    if (cvec == nullptr) {
        return false;
    }

    llvm::Constant *splatConst = cvec->getSplatValue();
    if (splatConst == nullptr) {
        return false;
    }

    llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(splatConst);
    if (ci == nullptr) {
        return false;
    }

    int64_t splatVal = ci->getSExtValue();
    *splat = (int)splatVal;
    return true;
}

static llvm::Value *lExtract248Scale(llvm::Value *splatOperand, int splatValue, llvm::Value *otherOperand,
                                     llvm::Value **result) {
    if (splatValue == 2 || splatValue == 4 || splatValue == 8) {
        *result = otherOperand;
        return LLVMInt32(splatValue);
    }
    // Even if we don't have a common scale by exactly 2, 4, or 8, we'll
    // see if we can pull out that much of the scale anyway; this may in
    // turn allow other optimizations later.
    for (int scale = 8; scale >= 2; scale /= 2) {
        llvm::Instruction *insertBefore = llvm::dyn_cast<llvm::Instruction>(*result);
        Assert(insertBefore != nullptr);

        if ((splatValue % scale) == 0) {
            // *result = otherOperand * splatOperand / scale;
            llvm::Value *splatScaleVec = (splatOperand->getType() == LLVMTypes::Int32VectorType)
                                             ? LLVMInt32Vector(scale)
                                             : LLVMInt64Vector(scale);
            llvm::Value *splatDiv = llvm::BinaryOperator::Create(llvm::Instruction::SDiv, splatOperand, splatScaleVec,
                                                                 "div", ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            *result = llvm::BinaryOperator::Create(llvm::Instruction::Mul, splatDiv, otherOperand, "mul",
                                                   ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
            return LLVMInt32(scale);
        }
    }
    return LLVMInt32(1);
}

/** The scale extraction of lExtractOffsetVector248Scale() for the operand of
    a zext: zext(x * s) is zext(x) * s only if x * s doesn't wrap, so only a
    nuw shl by 1, 2 or 3 or a nuw multiply by 2, 4 or 8 is looked through,
    without recursion or a partial scale. InstCombine turns the multiply of an
    unsigned index by the element size into a shl.

    A zext nneg is not looked through. nneg bounds only the product, not x
    (an unflagged mul i32 -1073741823, 4 wraps to 4), and the product
    narrows to a 32-bit index anyway (lOffsets32BitSafe()). Extracting a
    scale of a nuw or nsw product would be exact, but when the gather then
    becomes a load or store, lGSToLoadStore() rebuilds the offset as
    sext(x) * scale, which no longer matches the shl of neighboring
    accesses: the loop loses its scaled induction variable addressing
    (07_loop_unroll_varying).
 */
static llvm::Value *lExtractNUWScale(llvm::Value **vec) {
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(*vec);
    if (bop == nullptr) {
        return LLVMInt32(1);
    }
    int splat = 0;
    if (bop->getOpcode() == llvm::Instruction::Shl) {
        if (bop->hasNoUnsignedWrap() && lIsIntegerSplat(bop->getOperand(1), &splat) && splat >= 1 && splat <= 3) {
            *vec = bop->getOperand(0);
            return LLVMInt32(1 << splat);
        }
    } else if (bop->getOpcode() == llvm::Instruction::Mul && bop->hasNoUnsignedWrap()) {
        for (int i = 0; i < 2; ++i) {
            if (lIsIntegerSplat(bop->getOperand(i), &splat) && (splat == 2 || splat == 4 || splat == 8)) {
                *vec = bop->getOperand(1 - i);
                return LLVMInt32(splat);
            }
        }
    }
    return LLVMInt32(1);
}

/** Given a vector of integer offsets to a base pointer being used for a
    gather or a scatter, see if its root operation is a multiply by a
    vector of some value by all 2s/4s/8s.  If not, return nullptr.

    If it is return an i32 value of 2, 4, 8 from the function and modify
    *vec so that it points to the operand that is being multiplied by
    2/4/8.

    We go through all this trouble so that we can pass the i32 scale factor
    to the {gather,scatter}_base_offsets function as a separate scale
    factor for the offsets.  This in turn is used in a way so that the LLVM
    x86 code generator matches it to apply x86's free scale by 2x, 4x, or
    8x to one of two registers being added together for an addressing
    calculation.
 */
static llvm::Value *lExtractOffsetVector248Scale(llvm::Value **vec) {
    llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(*vec);
    if (cast != nullptr) {
        llvm::Value *castOp = cast->getOperand(0);
        llvm::Value *scale = nullptr;
        if (cast->getOpcode() == llvm::Instruction::ZExt) {
            if (cast->hasNonNeg()) {
                return LLVMInt32(1);
            }
            // zext(x * s) is zext(x) * s only if x * s doesn't wrap.
            scale = lExtractNUWScale(&castOp);
            if (castOp == cast->getOperand(0)) {
                return scale;
            }
        } else {
            scale = lExtractOffsetVector248Scale(&castOp);
        }
        if (scale == nullptr) {
            return nullptr;
        }

        // make a new cast instruction so that we end up with the right
        // type
        llvm::CastInst *newCast = llvm::CastInst::Create(cast->getOpcode(), castOp, cast->getType(), "offset_cast",
                                                         ISPC_INSERTION_POINT_INSTRUCTION(cast));
        *vec = newCast;
        return scale;
    }

    // If we don't have a binary operator, then just give up
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(*vec);
    if (bop == nullptr) {
        return LLVMInt32(1);
    }

    llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
    if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
        if (llvm::isa<llvm::ConstantAggregateZero>(op0)) {
            *vec = op1;
            return lExtractOffsetVector248Scale(vec);
        } else if (llvm::isa<llvm::ConstantAggregateZero>(op1)) {
            *vec = op0;
            return lExtractOffsetVector248Scale(vec);
        } else {
            llvm::Value *s0 = lExtractOffsetVector248Scale(&op0);
            llvm::Value *s1 = lExtractOffsetVector248Scale(&op1);
            if (s0 == s1) {
                *vec = llvm::BinaryOperator::Create(llvm::Instruction::Add, op0, op1, "new_add",
                                                    ISPC_INSERTION_POINT_INSTRUCTION(bop));
                return s0;
            } else {
                return LLVMInt32(1);
            }
        }
    } else if (bop->getOpcode() == llvm::Instruction::Mul) {
        // Check each operand for being one of the scale factors we care about.
        int splat = 0;
        if (lIsIntegerSplat(op0, &splat)) {
            return lExtract248Scale(op0, splat, op1, vec);
        } else if (lIsIntegerSplat(op1, &splat)) {
            return lExtract248Scale(op1, splat, op0, vec);
        } else {
            return LLVMInt32(1);
        }
    } else {
        return LLVMInt32(1);
    }
}

static bool lVectorIs32BitInts(llvm::Value *v) {
    int nElts = 0;
    int64_t elts[ISPC_MAX_NVEC];
    if (!LLVMExtractVectorInts(v, elts, &nElts)) {
        return false;
    }

    for (int i = 0; i < nElts; ++i) {
        if ((int32_t)elts[i] != elts[i]) {
            return false;
        }
    }

    return true;
}

/** Check to see if the two offset vectors can safely be represented with
    32-bit values.  If so, return true and update the pointed-to
    llvm::Value *s to be the 32-bit equivalents. */
static bool lOffsets32BitSafe(llvm::Value **variableOffsetPtr, llvm::Value **constOffsetPtr,
                              llvm::Instruction *insertBefore) {
    llvm::Value *variableOffset = *variableOffsetPtr;
    llvm::Value *constOffset = *constOffsetPtr;

    if (variableOffset->getType() != LLVMTypes::Int32VectorType) {
        if (auto *castInst = llvm::dyn_cast<llvm::CastInst>(variableOffset)) {
            auto opcode = castInst->getOpcode();
            // The 32-bit variants sign extend the offsets, so a zext is only
            // good if the sign bit is known to be clear. lExtractConstantOffset()
            // refused to split a zext nneg whose sum may wrap, like
            // zext nneg (x + INT32_MIN); without the zext, the exposed i32 sum
            // would be split later (lGSBaseOffsetsGetMoreConst()) into
            // sext(x) + sext(INT32_MIN), -2^32 instead of 0 for x = INT32_MIN.
            llvm::Value *narrow = castInst->getOperand(0);
            bool zextNNegOK = opcode == llvm::Instruction::ZExt && castInst->hasNonNeg() && !lHasConstantPart(narrow);
            if ((opcode == llvm::Instruction::SExt || zextNNegOK) && narrow->getType() == LLVMTypes::Int32VectorType) {

                // sext or zext nneg of a 32-bit vector -> the 32-bit vector is good
                variableOffset = castInst->getOperand(0);
            } else {
                return false;
            }
        } else if (lVectorIs32BitInts(variableOffset)) {
            // The only constant vector we should have here is a vector of
            // all zeros (i.e. a ConstantAggregateZero, but just in case,
            // do the more general check with lVectorIs32BitInts().
            variableOffset = new llvm::TruncInst(variableOffset, LLVMTypes::Int32VectorType,
                                                 llvm::Twine(variableOffset->getName()) + "_trunc",
                                                 ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
        } else {
            return false;
        }
    }

    if (constOffset->getType() != LLVMTypes::Int32VectorType) {
        if (lVectorIs32BitInts(constOffset)) {
            // Truncate them so we have a 32-bit vector type for them.
            constOffset = new llvm::TruncInst(constOffset, LLVMTypes::Int32VectorType,
                                              llvm::Twine(constOffset->getName()) + "_trunc",
                                              ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
        } else {
            // A zext of a 32-bit constant with the sign bit set, as split off
            // from an unsigned offset, doesn't fit.
            llvm::ZExtInst *zext = llvm::dyn_cast<llvm::ZExtInst>(constOffset);
            if (zext != nullptr && !zext->hasNonNeg() && !lVectorIs32BitInts(zext->getOperand(0))) {
                return false;
            }
            // Neither does a folded constant that lVectorIs32BitInts() rejected,
            // like the 2^31 of p[0x20000000 + programIndex] for a float p.
            int nElts = 0;
            int64_t elts[ISPC_MAX_NVEC];
            if (LLVMExtractVectorInts(constOffset, elts, &nElts)) {
                return false;
            }
            // Nor a constant part that folds to such a value, like the
            // 0x40000000 + 0x40000000 of (x + 0x40000000) + 0x40000000.
            const llvm::DataLayout &dl = insertBefore->getModule()->getDataLayout();
            if (llvm::Constant *folded = lFoldToConstant(constOffset, dl)) {
                if (!lConstantFits32(folded)) {
                    return false;
                }
            }
            // FIXME: otherwise we just assume that all constant offsets
            // can actually always fit into 32-bits...  (This could be
            // wrong, but it should be only in pretty esoteric cases).  We
            // make this assumption for now since we sometimes generate
            // constants that need constant folding before we really have a
            // constant vector out of them, and
            // llvm::ConstantFoldInstruction() doesn't seem to be doing
            // enough for us in some cases if we call it from here.
            constOffset = new llvm::TruncInst(constOffset, LLVMTypes::Int32VectorType,
                                              llvm::Twine(constOffset->getName()) + "_trunc",
                                              ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
        }
    }

    *variableOffsetPtr = variableOffset;
    *constOffsetPtr = constOffset;
    return true;
}

/** Check to see if the offset value is composed of a string of Adds,
    SExts, and Constant Vectors that are 32-bit safe.  Recursively
    explores the operands of Add instructions (as they might themselves
    be adds that eventually terminate in constant vectors or a SExt.)
 */

static bool lIs32BitSafeHelper(llvm::Value *v) {
    // handle Adds, SExts, Constant Vectors. A sum of 32-bit parts is not
    // known to fit in 32 bits (address-gap-narrow-sum.ispc), so zext nneg
    // isn't accepted here, unlike in lOffsets32BitSafe(): that would widen
    // this gap to the unsigned and the nonnegative indices.
    if (llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v)) {
        if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
            return lIs32BitSafeHelper(bop->getOperand(0)) && lIs32BitSafeHelper(bop->getOperand(1));
        }
        return false;
    } else if (llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(v)) {
        return sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType;
    } else {
        return lVectorIs32BitInts(v);
    }
}

/** Check to see if the single offset vector can safely be represented with
    32-bit values.  If so, return true and update the pointed-to
    llvm::Value * to be the 32-bit equivalent. */
static bool lOffsets32BitSafe(llvm::Value **offsetPtr, llvm::Instruction *insertBefore) {
    llvm::Value *offset = *offsetPtr;

    if (offset->getType() == LLVMTypes::Int32VectorType) {
        return true;
    }

    // The 32-bit native forms sign extend their index before applying the
    // hardware scale. A zext nneg has the same value as a sext on every
    // defined lane.
    auto *cast = llvm::dyn_cast<llvm::CastInst>(offset);
    if (cast != nullptr &&
        (cast->getOpcode() == llvm::Instruction::SExt ||
         (cast->getOpcode() == llvm::Instruction::ZExt && cast->hasNonNeg())) &&
        cast->getOperand(0)->getType() == LLVMTypes::Int32VectorType) {
        *offsetPtr = cast->getOperand(0);
        return true;
    } else if (lIs32BitSafeHelper(offset)) {
        // The only constant vector we should have here is a vector of
        // all zeros (i.e. a ConstantAggregateZero, but just in case,
        // do the more general check with lVectorIs32BitInts().

        // Alternatively, offset could be a sequence of adds terminating
        // in safe constant vectors or a SExt.
        *offsetPtr = new llvm::TruncInst(offset, LLVMTypes::Int32VectorType, llvm::Twine(offset->getName()) + "_trunc",
                                         ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
        return true;
    } else {
        return false;
    }
}

static llvm::CallInst *lGSToGSBaseOffsets(const MemoryAccess &access) {
    // Only the general pseudo gathers and scatters are converted. A prefetch
    // is left alone: no target has a vector prefetch instruction.
    if (!access.isPseudo() || access.form() != MemoryOpForm::General ||
        (access.kind() != MemoryOpKind::Gather && access.kind() != MemoryOpKind::Scatter)) {
        return nullptr;
    }
    // The emission below builds vectors of the target width. A call with
    // another lane count, which only hand-written IR has, is left alone;
    // RecoverBaseOffsets() itself accepts any lane count.
    if (access.laneCount() != static_cast<unsigned>(g->target->getVectorWidth())) {
        return nullptr;
    }
    llvm::CallInst *callInst = access.call;
    const bool isGather = access.kind() == MemoryOpKind::Gather;
    // The target's preference selects the base-offsets form, or the factored
    // form that the emulated implementations take; the offset width stays
    // that of the addresses unless the offsets are narrowed below.
    const bool native = isGather ? g->target->useGather() : g->target->useScatter();
    MemoryOpKey replacement = access.op->key;
    replacement.form = native ? MemoryOpForm::BaseOffsets : MemoryOpForm::Factored;
    MemoryOpKey narrowed = replacement;
    narrowed.offsetBits = 32;
    llvm::Module *M = callInst->getModule();
    llvm::Function *gatherScatterFunc = GetMemoryOpFunction(M, replacement);
    if (gatherScatterFunc == nullptr || (g->opt.force32BitAddressing && GetMemoryOpFunction(M, narrowed) == nullptr)) {
        // The replacement function this conversion may select is not
        // declared in this module (the frontend always declares them; only
        // hand-written IR lacks them). Which of the two widths is used is
        // only known after the recipe is materialized, so both must be
        // present rather than mutating the IR and then giving up.
        return nullptr;
    }

    // Try to describe the array of pointers as a single base pointer and
    // an array of offsets. The query is read-only; the recipe is
    // materialized only once the conversion is accepted, so a general
    // gather/scatter with varying base pointers leaves the IR unchanged and
    // is left as is.
    std::optional<BaseOffsetsRecipe> recipe = RecoverBaseOffsets(access);
    if (!recipe) {
        return nullptr;
    }
    // The recovered base is a numeric decomposition of the addresses, not a
    // proof of pointer provenance (#3983); the existing emission below treats
    // it as the legacy matcher's result.
    BaseOffsetsValues recovered = MaterializeBaseOffsets(*recipe, callInst);
    llvm::Value *basePtr = recovered.base;
    llvm::Value *offsetVector = recovered.offsets;

    // Cast the base pointer to a void *, since that's what the
    // __pseudo_*_base_offsets_* functions want.
    basePtr = new llvm::IntToPtrInst(basePtr, LLVMTypes::VoidPointerType, llvm::Twine(basePtr->getName()) + "_2void",
                                     ISPC_INSERTION_POINT_INSTRUCTION(callInst));
    LLVMCopyMetadata(basePtr, callInst);
    llvm::CallInst *newCall = nullptr;

    if (native) {

        // See if the offsets are scaled by 2, 4, or 8.  If so,
        // extract that scale factor and rewrite the offsets to remove
        // it.
        llvm::Value *offsetScale = lExtractOffsetVector248Scale(&offsetVector);

        // If we're doing 32-bit addressing on a 64-bit target, here we
        // will see if we can call one of the 32-bit variants of the pseudo
        // gather/scatter functions.
        if (g->opt.force32BitAddressing && lOffsets32BitSafe(&offsetVector, callInst)) {
            gatherScatterFunc = GetMemoryOpFunction(M, narrowed);
        }

        if (isGather) {
            llvm::Value *mask = access.mask();

            // Generate a new function call to the next pseudo gather
            // base+offsets instruction.  Note that we're passing a nullptr
            // llvm::Instruction to llvm::CallInst::Create; this means that
            // the instruction isn't inserted into a basic block and that
            // way we can then call ReplaceInstWithInst().
            newCall = LLVMCallInst(gatherScatterFunc, basePtr, offsetScale, offsetVector, mask,
                                   callInst->getName().str().c_str(), nullptr);
            LLVMCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        } else {
            llvm::Value *storeValue = access.value();
            llvm::Value *mask = access.mask();

            // Generate a new function call to the next pseudo scatter
            // base+offsets instruction.  See above for why passing nullptr
            // for the Instruction * is intended.
            newCall =
                LLVMCallInst(gatherScatterFunc, basePtr, offsetScale, offsetVector, storeValue, mask, "", nullptr);
            LLVMCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
    } else {
        // Try to decompose the offset vector into a compile time constant
        // component and a varying component.  The constant component is
        // passed as a separate parameter to the gather/scatter functions,
        // which in turn allows their implementations to end up emitting
        // x86 instructions with constant offsets encoded in them.
        llvm::Value *constOffset = nullptr;
        llvm::Value *variableOffset = nullptr;
        lExtractConstantOffset(offsetVector, &constOffset, &variableOffset, callInst);
        // A constant part that doesn't fit in 32 bits keeps the offset in the
        // 64-bit form, unsplit: a variable part the split built can have
        // wrapped at i32 width, which only the truncated constant offsets.
        if (constOffset != nullptr && g->opt.force32BitAddressing &&
            constOffset->getType() == LLVMTypes::Int64VectorType) {
            llvm::Constant *folded = lFoldToConstant(constOffset, callInst->getModule()->getDataLayout());
            if (folded != nullptr && !lConstantFits32(folded)) {
                constOffset = nullptr;
                variableOffset = offsetVector;
            }
        }
        if (constOffset == nullptr) {
            constOffset = LLVMIntAsType(0, offsetVector->getType());
        }
        if (variableOffset == nullptr) {
            variableOffset = LLVMIntAsType(0, offsetVector->getType());
        }

        // See if the varying component is scaled by 2, 4, or 8.  If so,
        // extract that scale factor and rewrite variableOffset to remove
        // it.  (This also is pulled out so that we can match the scales by
        // 2/4/8 offered by x86 addressing operators.)
        llvm::Value *offsetScale = lExtractOffsetVector248Scale(&variableOffset);

        // If we're doing 32-bit addressing on a 64-bit target, here we
        // will see if we can call one of the 32-bit variants of the pseudo
        // gather/scatter functions.
        if (g->opt.force32BitAddressing && lOffsets32BitSafe(&variableOffset, &constOffset, callInst)) {
            gatherScatterFunc = GetMemoryOpFunction(M, narrowed);
        }

        if (isGather) {
            llvm::Value *mask = access.mask();

            // Generate a new function call to the next pseudo gather
            // base+offsets instruction.  Note that we're passing a nullptr
            // llvm::Instruction to llvm::CallInst::Create; this means that
            // the instruction isn't inserted into a basic block and that
            // way we can then call ReplaceInstWithInst().
            newCall = LLVMCallInst(gatherScatterFunc, basePtr, variableOffset, offsetScale, constOffset, mask,
                                   callInst->getName().str().c_str(), nullptr);
            LLVMCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        } else {
            llvm::Value *storeValue = access.value();
            llvm::Value *mask = access.mask();

            // Generate a new function call to the next pseudo scatter
            // base+offsets instruction.  See above for why passing nullptr
            // for the Instruction * is intended.
            newCall = LLVMCallInst(gatherScatterFunc, basePtr, variableOffset, offsetScale, constOffset, storeValue,
                                   mask, "", nullptr);
            LLVMCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
        }
    }
    return newCall;
}

/** Returns true if v is root or one of the operands that
    lExtractConstantOffset() looks through in root, i.e. a value of the
    original expression rather than one the split built. */
static bool lIsOriginalPart(llvm::Value *v, llvm::Value *root, int depth = 0) {
    if (v == root) {
        return true;
    }
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(root);
    if (inst == nullptr || depth > 16 || !(llvm::isa<llvm::BinaryOperator>(inst) || llvm::isa<llvm::CastInst>(inst))) {
        return false;
    }
    for (llvm::Value *op : inst->operands()) {
        if (lIsOriginalPart(v, op, depth + 1)) {
            return true;
        }
    }
    return false;
}

/** Returns constOffset * scale + delta as an i32 vector constant, with each
    element sign extended and the arithmetic done in i64, or nullptr if the
    parts don't fold to constants or a lane doesn't fit in i32.

    If exact, the offset constOffset was split from is all nsw, and the
    constant part is evaluated exactly (lEvalExact()). If the narrow constant
    must also not have wrapped (sameAsNarrow), its value as the IR computes
    it must be the exact one. Otherwise the constant part is taken as the IR
    computes it. */
static llvm::Constant *lScaledDelta32(llvm::Value *constOffset, bool exact, bool sameAsNarrow, int64_t scale,
                                      llvm::Value *delta, const llvm::DataLayout &dl) {
    llvm::Constant *c = lFoldToConstant(constOffset, dl);
    llvm::Constant *d = lFoldToConstant(delta, dl);
    if (c == nullptr || d == nullptr) {
        return nullptr;
    }
    unsigned n = llvm::cast<llvm::FixedVectorType>(constOffset->getType())->getNumElements();
    std::vector<llvm::Constant *> lanes;
    for (unsigned i = 0; i < n; ++i) {
        llvm::ConstantInt *ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(c->getAggregateElement(i));
        llvm::ConstantInt *di = llvm::dyn_cast_or_null<llvm::ConstantInt>(d->getAggregateElement(i));
        if (ci == nullptr || di == nullptr || ci->getBitWidth() > 64) {
            return nullptr;
        }
        llvm::APInt value = ci->getValue().sext(64);
        if (exact) {
            llvm::APInt exactValue;
            if (!lEvalExact(constOffset, i, &exactValue, dl) || (sameAsNarrow && exactValue != value)) {
                return nullptr;
            }
            value = exactValue;
        }
        bool overflow = false;
        value = value.smul_ov(llvm::APInt(64, scale, /*isSigned*/ true), overflow);
        if (!overflow) {
            value = value.sadd_ov(di->getValue().sext(64), overflow);
        }
        if (overflow || !value.isSignedIntN(32)) {
            return nullptr;
        }
        lanes.push_back(LLVMInt32((int32_t)value.getSExtValue()));
    }
    return llvm::ConstantVector::get(lanes);
}

/** Try to improve the decomposition between compile-time constant and
    compile-time unknown offsets in calls to the __pseudo_*_base_offsets*
    functions. After other optimizations have run, we will sometimes be
    able to pull more terms out of the unknown part and add them into the
    compile-time-known part.
 */
static llvm::CallInst *lGSBaseOffsetsGetMoreConst(const MemoryAccess &access) {
    // Only the factored pseudo forms carry a separate constant offset. (The
    // legacy table also listed the non-factored base-offset forms, whose
    // uniform scale in that operand position never decomposed.)
    if (!access.isPseudo() || access.form() != MemoryOpForm::Factored ||
        (access.kind() != MemoryOpKind::Gather && access.kind() != MemoryOpKind::Scatter)) {
        return nullptr;
    }
    llvm::CallInst *callInst = access.call;

    // Grab the old variable offset
    llvm::Value *origVariableOffset = access.offsets();

    // If it's zero, we're done.  Don't go and think that we're clever by
    // adding these zeros to the constant offsets.
    if (llvm::isa<llvm::ConstantAggregateZero>(origVariableOffset)) {
        return nullptr;
    }

    // With 64-bit pointers, the lowering of the 32-bit form adds sext(delta)
    // to sext(offsets) * scale in i64 (__gather_elt32_*).
    const bool offsets32 = origVariableOffset->getType() == LLVMTypes::Int32VectorType && !g->target->is32Bit();

    // Try to decompose the old variable offset
    llvm::Value *constOffset = nullptr;
    llvm::Value *variableOffset = nullptr;
    lExtractConstantOffset(origVariableOffset, &constOffset, &variableOffset, callInst);

    // No luck
    if (constOffset == nullptr) {
        return nullptr;
    }

    // Total luck: everything could be moved to the constant offset
    const bool allConstant = variableOffset == nullptr;
    if (allConstant) {
        variableOffset = LLVMIntAsType(0, origVariableOffset->getType());
    }

    // We need to scale the value we add to the constant offset by the
    // 2/4/8 scale for the variable offset, if present.
    llvm::ConstantInt *varScale = llvm::dyn_cast<llvm::ConstantInt>(access.scale());
    Assert(varScale != nullptr);

    // The scale is signed, as in the lowering (__gather_elt*_*).
    const int64_t scale = varScale->getSExtValue();

    // The new i32 delta must not wrap either: a constant 0x40000000 with
    // scale 4 is 4 GiB, not 0.
    if (offsets32) {
        // If the offset is all nsw, the constant part is evaluated exactly:
        // regrouped at i32 width, 0x40000000 + 0x40000000 wraps to INT32_MIN.
        // Then variable + constant is the offset if the variable part is a
        // value of the original offset. A variable part the split built,
        // like x + y of ((x + 0x40000000) + 0x40000000) + y, is computed at
        // i32 width; the constant must then not have wrapped either.
        bool exact = lSplitIsNSW(origVariableOffset);
        bool sameAsNarrow = !allConstant && !lIsOriginalPart(variableOffset, origVariableOffset);
        llvm::Constant *delta = lScaledDelta32(constOffset, exact, sameAsNarrow, scale, access.delta(),
                                               callInst->getModule()->getDataLayout());
        if (delta == nullptr) {
            return nullptr;
        }
        callInst->setArgOperand(access.op->layout.offsets, variableOffset);
        callInst->setArgOperand(access.op->layout.delta, delta);
        return callInst;
    }

    llvm::Value *scaleSmear = nullptr;
    if (origVariableOffset->getType() == LLVMTypes::Int64VectorType) {
        scaleSmear = LLVMInt64Vector(scale);
    } else {
        scaleSmear = LLVMInt32Vector((int32_t)scale);
    }

    constOffset = llvm::BinaryOperator::Create(llvm::Instruction::Mul, constOffset, scaleSmear, constOffset->getName(),
                                               ISPC_INSERTION_POINT_INSTRUCTION(callInst));

    // And add the additional offset to the original constant offset
    constOffset = llvm::BinaryOperator::Create(llvm::Instruction::Add, constOffset, access.delta(),
                                               access.delta()->getName(), ISPC_INSERTION_POINT_INSTRUCTION(callInst));

    // Finally, update the values of the operands to the gather/scatter
    // function.
    callInst->setArgOperand(access.op->layout.offsets, variableOffset);
    callInst->setArgOperand(access.op->layout.delta, constOffset);

    return callInst;
}

static llvm::Value *lComputeCommonPointer(llvm::Value *base, llvm::Type *baseType, llvm::Value *offsets,
                                          llvm::Instruction *insertBefore) {
    llvm::Value *firstOffset = LLVMExtractFirstVectorElement(offsets);
    Assert(firstOffset != nullptr);

    return LLVMGEPInst(base, baseType, firstOffset, "ptr", insertBefore);
}

/** Returns true if x, a vector, has the same value in all lanes, like a splat,
    a broadcast or the phi of a uniform loop counter.
 */
static bool lIsUniformVector(llvm::Value *x) {
    if (llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(x)) {
        return c->getSplatValue() != nullptr;
    }
    return llvm::isa<llvm::Instruction>(x) && LLVMVectorValuesAllEqual(x);
}

/** Returns true if x is a uniform value u plus lane constants c that can't
    wrap relative to lane 0: an add nuw with c smallest in lane 0 or an or
    disjoint with c zero in lane 0, possibly shifted or multiplied by a
    constant with nuw, like zext(shl nuw (or disjoint %i, <0, 1, ...>), 1)
    that InstCombine makes of the offset of a signed loop counter. Lane 0 is
    then never poison, and it is the smallest lane. So if a lane that is on
    doesn't wrap, lane 0 doesn't either.
 */
static bool lIsNoWrapUniformPlusLanes(llvm::Value *x) {
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(x);
    if (bop == nullptr) {
        return false;
    }
    llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
    switch (bop->getOpcode()) {
    case llvm::Instruction::Shl:
    case llvm::Instruction::Mul:
        return bop->hasNoUnsignedWrap() && llvm::isa<llvm::Constant>(op1) && lIsUniformVector(op1) &&
               lIsNoWrapUniformPlusLanes(op0);
    case llvm::Instruction::Add:
    case llvm::Instruction::Or: {
        bool isAdd = bop->getOpcode() == llvm::Instruction::Add;
        if ((isAdd && !bop->hasNoUnsignedWrap()) || (!isAdd && !IsOrEquivalentToAdd(bop))) {
            return false;
        }
        if (lIsUniformVector(op1)) {
            std::swap(op0, op1);
        }
        int nElts = 0;
        int64_t elts[ISPC_MAX_NVEC];
        // Only a broadcast or a phi, like that of a uniform loop counter, for u:
        // LLVMVectorValuesAllEqual() also accepts vectors with some lanes poison.
        bool isBroadcastOrPhi = llvm::getSplatValue(op0) != nullptr || llvm::isa<llvm::PHINode>(op0);
        if (!isBroadcastOrPhi || !lIsUniformVector(op0) || !llvm::isa<llvm::Constant>(op1) ||
            !LLVMExtractVectorInts(op1, elts, &nElts)) {
            return false;
        }
        // The elements are zero extended, so this compares them as unsigned.
        for (int i = 0; i < nElts; ++i) {
            if (isAdd ? elts[i] < elts[0] : elts[0] != 0) {
                return false;
            }
        }
        return true;
    }
    default:
        return false;
    }
}

/** Returns true if the lanes of x, the operand of a zext, may wrap relative to
    each other, like x = j - 1 for an unsigned j = b + programIndex, whose lane
    0 is 0xFFFFFFFF for b = 0: zext(x) is then linear only modulo 2^32, and the
    offset of lane 0 is 2^32 times the element size away from those of the
    other lanes, or poison if it was multiplied with nuw. A uniform
    contribution like b - d may be close to 2^32 even for small b and d, so
    only a uniform or constant x, or one that lIsNoWrapUniformPlusLanes()
    accepts, is known not to.
 */
static bool lZExtOperandMayWrap(llvm::Value *x) {
    if (lIsUniformVector(x)) {
        return false;
    }
    // Constant parts split off by lExtractConstantOffset() may not be folded.
    // Folding ignores nuw and nsw, so only instructions without them.
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(x);
    if (inst != nullptr && !inst->hasPoisonGeneratingFlags()) {
        if (llvm::Constant *folded = llvm::ConstantFoldInstruction(inst, inst->getModule()->getDataLayout())) {
            x = folded;
        }
    }
    if (!llvm::isa<llvm::Constant>(x)) {
        return !lIsNoWrapUniformPlusLanes(x);
    }
    int nElts = 0;
    int64_t elts[ISPC_MAX_NVEC];
    if (!LLVMExtractVectorInts(x, elts, &nElts)) {
        return !llvm::isa<llvm::ConstantAggregateZero>(x);
    }
    for (int i = 0; i < nElts; ++i) {
        if (elts[i] < 0 || elts[i] > INT32_MAX) {
            return true;
        }
    }
    return false;
}

/** Returns true if the offsets in vec are computed from a zext whose operand
    may wrap (see lZExtOperandMayWrap()). Then the offsets may be linear only
    modulo 2^32, and the offset of a lane that is off may be 4 GB away. For
    example, p[j - 1] in "if (j > 0)" with an unsigned j = programIndex has
    the index 0xFFFFFFFF in lane 0, whose offset is poison or 16 GB away for
    a float. That the lanes that are on don't straddle the wrap themselves
    isn't checked: LLVMVectorIsLinear() looks through the zext. Phis are
    followed, as LLVMVectorIsLinear() does. Past a depth of 8, a zext is
    assumed.
 */
static bool lHasWrappingZExt(llvm::Value *vec, llvm::SmallPtrSetImpl<llvm::Value *> &visited, int depth = 0) {
    if (depth > 8) {
        return true;
    }
    if (llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(vec)) {
        if (cast->getOpcode() == llvm::Instruction::ZExt && !cast->hasNonNeg()) {
            return lZExtOperandMayWrap(cast->getOperand(0));
        }
        return lHasWrappingZExt(cast->getOperand(0), visited, depth + 1);
    }
    if (llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec)) {
        return lHasWrappingZExt(bop->getOperand(0), visited, depth + 1) ||
               lHasWrappingZExt(bop->getOperand(1), visited, depth + 1);
    }
    if (llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(vec)) {
        if (!visited.insert(phi).second) {
            return false;
        }
        for (llvm::Value *incoming : phi->incoming_values()) {
            if (lHasWrappingZExt(incoming, visited, depth + 1)) {
                return true;
            }
        }
    }
    return false;
}

/** Like lComputeCommonPointer(), but computes the pointer from the offset of
    the first lane that is on (or of lane 0 if none is): offsets that are
    linear with the given stride for the lanes that are on give the same
    pointer for each of them.
 */
static llvm::Value *lComputeCommonPointerFromFirstOnLane(llvm::Value *base, llvm::Type *baseType, llvm::Value *offsets,
                                                         llvm::Value *mask, int stride,
                                                         llvm::Instruction *insertBefore) {
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(offsets->getType());
    Assert(vt != nullptr);
    int nElts = vt->getNumElements();

    llvm::Value *on = mask;
    if (!mask->getType()->getScalarType()->isIntegerTy(1)) {
        on = new llvm::ICmpInst(ISPC_INSERTION_POINT_INSTRUCTION(insertBefore), llvm::CmpInst::ICMP_NE, mask,
                                llvm::Constant::getNullValue(mask->getType()), "on");
    }
    llvm::IntegerType *bitsType = llvm::IntegerType::get(*g->ctx, nElts);
    llvm::Value *bits = new llvm::BitCastInst(on, bitsType, "on_bits", ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
    llvm::Function *cttz =
        llvm::Intrinsic::getOrInsertDeclaration(insertBefore->getModule(), llvm::Intrinsic::cttz, {bitsType});
    llvm::Value *lane =
        llvm::CallInst::Create(cttz, {bits, LLVMFalse}, "first_on", ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
    // cttz gives nElts if no lane is on.
    lane = llvm::BinaryOperator::Create(llvm::Instruction::And, lane, llvm::ConstantInt::get(bitsType, nElts - 1),
                                        "first_on_lane", ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
    llvm::Type *offsetType = vt->getElementType();
    lane = llvm::CastInst::CreateIntegerCast(lane, offsetType, false, "first_on_lane_cast",
                                             ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));

    llvm::Value *laneOffset =
        llvm::ExtractElementInst::Create(offsets, lane, "lane_offset", ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
    llvm::Value *laneDelta =
        llvm::BinaryOperator::Create(llvm::Instruction::Mul, lane, llvm::ConstantInt::get(offsetType, stride),
                                     "lane_delta", ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
    llvm::Value *firstOffset = llvm::BinaryOperator::Create(
        llvm::Instruction::Sub, laneOffset, laneDelta, "first_offset", ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));

    return LLVMGEPInst(base, baseType, firstOffset, "ptr", insertBefore);
}

/** For negative stride access patterns (e.g., dst[size-1-i]), we need to compute
    the pointer from the last element of the offset vector (which has the lowest address).
 */
static llvm::Value *lComputeCommonPointerForNegativeStride(llvm::Value *base, llvm::Type *baseType,
                                                           llvm::Value *offsets, llvm::Instruction *insertBefore) {
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(offsets->getType());
    Assert(vt != nullptr);
    int lastIdx = vt->getNumElements() - 1;

    llvm::Value *lastOffset = llvm::ExtractElementInst::Create(offsets, LLVMInt32(lastIdx), "last_offset",
                                                               ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));

    return LLVMGEPInst(base, baseType, lastOffset, "ptr_backward", insertBefore);
}

/** Create a reversed vector using shuffle. For a vector {a,b,c,d,e,f,g,h},
    returns {h,g,f,e,d,c,b,a}. This is used to transform backward memory
    access patterns into contiguous accesses.
 */
static llvm::Instruction *lReverseVector(llvm::Value *v, llvm::Instruction *insertBefore) {
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
    Assert(vt != nullptr);
    int numElems = vt->getNumElements();

    // Create reverse shuffle mask: {n-1, n-2, ..., 1, 0}
    llvm::SmallVector<int, ISPC_MAX_NVEC> reverseMask;
    for (int i = numElems - 1; i >= 0; --i) {
        reverseMask.push_back(i);
    }

    return new llvm::ShuffleVectorInst(v, llvm::PoisonValue::get(vt), reverseMask, "reversed",
                                       ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
}

static llvm::Constant *lGetOffsetScaleVec(llvm::Value *offsetScale, llvm::Type *vecType) {
    llvm::ConstantInt *offsetScaleInt = llvm::dyn_cast<llvm::ConstantInt>(offsetScale);
    Assert(offsetScaleInt != nullptr);
    // The lowerings sign extend the i32 scale (__gather_elt32_* and
    // __gather_elt64_* in builtins/util.m4); a factored scale may be negative.
    int64_t scaleValue = offsetScaleInt->getSExtValue();

    std::vector<llvm::Constant *> scales;
    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
        if (vecType == LLVMTypes::Int64VectorType) {
            scales.push_back(LLVMInt64(scaleValue));
        } else {
            Assert(vecType == LLVMTypes::Int32VectorType);
            scales.push_back(LLVMInt32((int32_t)scaleValue));
        }
    }
    return llvm::ConstantVector::get(scales);
}

/** With 64-bit pointers, the x86 lowering of a 32-bit-index form sign
    extends the offsets and the delta before it scales and adds them
    (__gather_elt32_* in builtins/util.m4), so offsets * scale + delta may
    exceed i32. Returns v sign extended to i64 in that case, so that the
    load/store computes the same address; otherwise v itself. */
static llvm::Value *lExtendToIndexWidth(llvm::Value *v, llvm::Value *base, llvm::Instruction *insertBefore) {
    const llvm::DataLayout &dl = insertBefore->getModule()->getDataLayout();
    if (v->getType() != LLVMTypes::Int32VectorType ||
        dl.getPointerSizeInBits(base->getType()->getPointerAddressSpace()) != 64) {
        return v;
    }
    if (llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(v)) {
        if (llvm::Constant *folded =
                llvm::ConstantFoldCastOperand(llvm::Instruction::SExt, c, LLVMTypes::Int64VectorType, dl)) {
            return folded;
        }
    }
    return llvm::CastInst::Create(llvm::Instruction::SExt, v, LLVMTypes::Int64VectorType, "",
                                  ISPC_INSERTION_POINT_INSTRUCTION(insertBefore));
}

/** After earlier optimization passes have run, we are sometimes able to
    determine that gathers/scatters are actually accessing memory in a more
    regular fashion and then change the operation to something simpler and
    more efficient.  For example, if all of the lanes in a gather are
    reading from the same location, we can instead do a scalar load and
    broadcast.  This pass examines gathers and scatters and tries to
    simplify them if at all possible.

    @todo Currently, this only looks for all program instances going to the
    same location and all going to a linear sequence of locations in
    memory.  There are a number of other cases that might make sense to
    look for, including things that could be handled with a vector load +
    shuffle or things that could be handled with hybrids of e.g. 2 4-wide
    vector loads with AVX, etc.
*/
static llvm::Instruction *lGSToLoadStore(const MemoryAccess &access) {
    // Only a pseudo base-offset or factored gather/scatter is a candidate;
    // its operand layout follows from the callee, not from the target.
    if (!access.isPseudo() || (access.form() != MemoryOpForm::BaseOffsets && access.form() != MemoryOpForm::Factored) ||
        (access.kind() != MemoryOpKind::Gather && access.kind() != MemoryOpKind::Scatter)) {
        return nullptr;
    }
    llvm::ConstantInt *scale = llvm::dyn_cast<llvm::ConstantInt>(access.scale());
    if (scale == nullptr) {
        return nullptr;
    }
    llvm::CallInst *callInst = access.call;
    const bool isGather = access.kind() == MemoryOpKind::Gather;
    // Pseudo gather/scatter base pointers are i8 pointers, e.g.
    // @__pseudo_gather_base_offsets32_i8(ptr, i32, <WIDTH x i32>, <WIDTH x MASK>)
    llvm::Type *baseType = LLVMTypes::Int8Type;
    llvm::Type *scalarType = access.elementType();
    // The masked load/store of the same element type that replaces the call.
    auto maskedFunc = [&](MemoryOpKind kind, MemoryOpFlavor flavor) {
        MemoryOpKey key;
        key.kind = kind;
        key.form = MemoryOpForm::Pointer;
        key.element = access.element();
        key.flavor = flavor;
        return GetMemoryOpFunction(callInst->getModule(), key);
    };

    SourcePos pos;
    access.sourcePos(&pos);

    llvm::Value *base = access.base();
    llvm::Value *fullOffsets = nullptr;
    llvm::Value *storeValue = access.value();
    llvm::Value *mask = access.mask();
    if (access.form() == MemoryOpForm::Factored) {
        llvm::Value *varyingOffsets = lExtendToIndexWidth(access.offsets(), base, callInst);
        llvm::Value *offsetScale = access.scale();
        llvm::Value *constOffsets = lExtendToIndexWidth(access.delta(), base, callInst);

        // Compute the full offset vector: offsetScale * varyingOffsets + constOffsets
        llvm::Constant *offsetScaleVec = lGetOffsetScaleVec(offsetScale, varyingOffsets->getType());

        llvm::BinaryOperator *scaledVarying =
            llvm::BinaryOperator::Create(llvm::Instruction::Mul, offsetScaleVec, varyingOffsets, "scaled_varying",
                                         ISPC_INSERTION_POINT_INSTRUCTION(callInst));
        llvm::BinaryOperator *sum =
            llvm::BinaryOperator::Create(llvm::Instruction::Add, scaledVarying, constOffsets, "varying+const_offsets",
                                         ISPC_INSERTION_POINT_INSTRUCTION(callInst));
        if (varyingOffsets != access.offsets()) {
            // Sign extended i32 values: |offset * scale| < 2^63 and
            // |offset * scale + delta| < 2^63, so neither wraps.
            scaledVarying->setHasNoSignedWrap(true);
            sum->setHasNoSignedWrap(true);
        }
        fullOffsets = sum;
    } else {
        llvm::Value *offsetScale = access.scale();
        llvm::Value *offsets = lExtendToIndexWidth(access.offsets(), base, callInst);

        // When offsetScale is 1, skip the multiplication to preserve linearity detection
        if (scale->isOne()) {
            fullOffsets = offsets;
        } else {
            llvm::Value *offsetScaleVec = lGetOffsetScaleVec(offsetScale, offsets->getType());
            llvm::BinaryOperator *scaled =
                llvm::BinaryOperator::Create(llvm::Instruction::Mul, offsetScaleVec, offsets, "scaled_offsets",
                                             ISPC_INSERTION_POINT_INSTRUCTION(callInst));
            if (offsets != access.offsets()) {
                // A sign extended i32 offset times the scale doesn't wrap in i64.
                scaled->setHasNoSignedWrap(true);
            }
            fullOffsets = scaled;
        }
    }

    Debug(SourcePos(), "GSToLoadStore: %s.", fullOffsets->getName().str().c_str());

    auto callName = callInst->getName();
    if (LLVMVectorValuesAllEqual(fullOffsets)) {
        // If all the offsets are equal, then compute the single
        // pointer they all represent based on the first one of them
        // (arbitrarily).
        if (isGather) {
            // A gather with everyone going to the same location is
            // handled as a scalar load and broadcast across the lanes.
            Debug(pos, "Transformed gather to scalar load and broadcast!");
            llvm::Value *ptr = lComputeCommonPointer(base, baseType, fullOffsets, callInst);

            LLVMCopyMetadata(ptr, callInst);
            Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
            llvm::Value *scalarValue =
                new llvm::LoadInst(scalarType, ptr, callInst->getName(), ISPC_INSERTION_POINT_INSTRUCTION(callInst));

            // Generate the following sequence:
            //   %name123 = insertelement <4 x i32> poison, i32 %val, i32 0
            //   %name124 = shufflevector <4 x i32> %name123, <4 x i32> poison,
            //                                              <4 x i32> zeroinitializer
            llvm::Value *undef1Value = llvm::PoisonValue::get(callInst->getType());
            llvm::Value *undef2Value = llvm::PoisonValue::get(callInst->getType());
            llvm::Value *insertVec = llvm::InsertElementInst::Create(undef1Value, scalarValue, LLVMInt32(0), callName,
                                                                     ISPC_INSERTION_POINT_INSTRUCTION(callInst));
            llvm::FixedVectorType *FVT = llvm::dyn_cast<llvm::FixedVectorType>(callInst->getType());
            if (!FVT) {
                Error(pos, "Unable to cast return type of \"%s\" to FixedVectorType", callName.str().c_str());
                return nullptr;
            }
            llvm::Value *zeroMask =
                llvm::ConstantVector::getSplat(llvm::ElementCount::get(FVT->getNumElements(), false),
                                               llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
            llvm::Instruction *shufInst = new llvm::ShuffleVectorInst(insertVec, undef2Value, zeroMask, callName);

            LLVMCopyMetadata(shufInst, callInst);
            llvm::ReplaceInstWithInst(callInst, shufInst);
            return shufInst;
        } else {
            // A scatter with everyone going to the same location is
            // undefined or poison (if there's more than one program instance in
            // the gang).  Issue a warning.
            if (g->target->getVectorWidth() > 1) {
                Warning(pos, "Undefined behavior: all program instances are "
                             "writing to the same location!");
            }

            // We could do something similar to the gather case, where
            // we arbitrarily write one of the values, but we need to
            // a) check to be sure the mask isn't all off and b) pick
            // the value from an executing program instance in that
            // case.  We'll just let a bunch of the program instances
            // do redundant writes, since this isn't important to make
            // fast anyway...
            return nullptr;
        }
    } else {
        int step = access.elementBytes();

        // Check for positive stride (forward access pattern)
        if (step > 0 && LLVMVectorIsLinear(fullOffsets, step)) {
            // We have a linear sequence of memory locations being accessed
            // starting with the location given by the offset from
            // offsetElements[0], with stride of 4 or 8 bytes (for 32 bit
            // and 64 bit gather/scatters, respectively.
            llvm::Instruction *newCall = nullptr;
            // Lanes that are off may have offsets that wrapped. Then the
            // pointer is computed from a lane that is on.
            llvm::SmallPtrSet<llvm::Value *, 4> visited;
            bool fromOnLane =
                GetMaskStatusFromValue(mask) != MaskStatus::all_on && lHasWrappingZExt(fullOffsets, visited);

            if (isGather) {
                llvm::Value *ptr =
                    fromOnLane ? lComputeCommonPointerFromFirstOnLane(base, baseType, fullOffsets, mask, step, callInst)
                               : lComputeCommonPointer(base, baseType, fullOffsets, callInst);
                LLVMCopyMetadata(ptr, callInst);
                Debug(pos, "Transformed gather to unaligned vector load!");
                bool doBlendLoad = false;
#ifdef ISPC_XE_ENABLED
                doBlendLoad = g->target->isXeTarget() && g->opt.enableXeUnsafeMaskedLoad;
#endif
                llvm::Function *loadFunc =
                    maskedFunc(MemoryOpKind::MaskedLoad, doBlendLoad ? MemoryOpFlavor::Blend : MemoryOpFlavor::Lowered);
                if (loadFunc == nullptr) {
                    // Not declared in this module (hand-written IR).
                    return nullptr;
                }
                newCall = LLVMCallInst(loadFunc, ptr, mask, llvm::Twine(ptr->getName()) + "_masked_load");
            } else {
                Debug(pos, "Transformed scatter to unaligned vector store!");
                llvm::Function *storeFunc = maskedFunc(MemoryOpKind::MaskedStore, MemoryOpFlavor::Pseudo);
                if (storeFunc == nullptr) {
                    // Not declared in this module (hand-written IR).
                    return nullptr;
                }
                llvm::Value *ptr =
                    fromOnLane ? lComputeCommonPointerFromFirstOnLane(base, baseType, fullOffsets, mask, step, callInst)
                               : lComputeCommonPointer(base, baseType, fullOffsets, callInst);
                newCall = LLVMCallInst(storeFunc, ptr, storeValue, mask, "");
            }
            LLVMCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
            return newCall;
        }

        // Check for negative stride (backward access pattern like dst[size-1-i])
        // This transforms scatter to: reverse data + contiguous store
        // And gather to: contiguous load + reverse result
        if (step > 0 && LLVMVectorIsLinear(fullOffsets, -step)) {
            // Phase 1 safety check: Only optimize when mask is all-on to avoid
            // partial mask complexity. For partial masks, the base pointer
            // calculation becomes complex since we'd need to find the last
            // active lane dynamically.
            MaskStatus maskStatus = GetMaskStatusFromValue(mask);
            if (maskStatus != MaskStatus::all_on) {
                return nullptr; // Fall back to scatter for non-all-on masks
            }

#ifdef ISPC_XE_ENABLED
            // Skip Xe targets until we verify shuffle support
            if (g->target->isXeTarget()) {
                return nullptr;
            }
#endif

            llvm::Instruction *newCall = nullptr;
            llvm::Function *memFunc = isGather ? maskedFunc(MemoryOpKind::MaskedLoad, MemoryOpFlavor::Lowered)
                                               : maskedFunc(MemoryOpKind::MaskedStore, MemoryOpFlavor::Pseudo);
            if (memFunc == nullptr) {
                return nullptr; // not declared in this module (hand-written IR)
            }

            if (isGather) {
                // For backward gather: load from lowest address, then reverse
                llvm::Value *ptr = lComputeCommonPointerForNegativeStride(base, baseType, fullOffsets, callInst);
                LLVMCopyMetadata(ptr, callInst);
                Debug(pos, "Transformed backward gather to load + reverse!");

                newCall = LLVMCallInst(memFunc, ptr, mask, llvm::Twine(ptr->getName()) + "_masked_load");
                newCall->insertBefore(ISPC_INSERTION_POINT_INSTRUCTION(callInst));
                LLVMCopyMetadata(newCall, callInst);

                // Reverse the loaded vector to restore correct lane order.
                // Insert after newCall, replace uses, then erase callInst.
                llvm::Instruction *reversed = lReverseVector(newCall, callInst);
                LLVMCopyMetadata(reversed, callInst);
                callInst->replaceAllUsesWith(reversed);
                callInst->eraseFromParent();
                return reversed;
            } else {
                // For backward scatter: reverse data, then store to lowest address
                Debug(pos, "Transformed backward scatter to reverse + store!");

                llvm::Value *ptr = lComputeCommonPointerForNegativeStride(base, baseType, fullOffsets, callInst);
                LLVMCopyMetadata(ptr, callInst);

                // Reverse the data vector before storing
                llvm::Value *reversedData = lReverseVector(storeValue, callInst);
                LLVMCopyMetadata(reversedData, callInst);

                newCall = LLVMCallInst(memFunc, ptr, reversedData, mask, "");
                LLVMCopyMetadata(newCall, callInst);
                llvm::ReplaceInstWithInst(callInst, newCall);
                return newCall;
            }
        }

        return nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////
// MaskedStoreOptPass

/** Masked stores are generally more complex than regular stores; for
    example, they require multiple instructions to simulate under SSE.
    This optimization detects cases where masked stores can be replaced
    with regular stores or removed entirely, for the cases of an 'all on'
    mask and an 'all off' mask, respectively.
*/
static llvm::Value *lImproveMaskedStore(const MemoryAccess &access) {
    // A masked store of any flavor; it is assumed aligned to its element.
    if (access.kind() != MemoryOpKind::MaskedStore) {
        return nullptr;
    }
    llvm::CallInst *callInst = access.call;

    // Got one; grab the operands
    llvm::Value *lvalue = access.base();
    llvm::Value *rvalue = access.value();
    llvm::Value *mask = access.mask();
    int align = access.naturalAlignment();

    // The legacy mask status chooses values for undef lanes; it is kept here
    // to preserve the existing codegen (MemoryAccess::maskStatus() reports
    // only proven statuses).
    MaskStatus maskStatus = GetMaskStatusFromValue(mask);
    if (maskStatus == MaskStatus::all_off) {
        // Zero mask - no-op, so remove the store completely.  (This
        // may in turn lead to being able to optimize out instructions
        // that compute the rvalue...)
        callInst->eraseFromParent();
        // Return some fake poison value to signal that we did transformation.
        return llvm::PoisonValue::get(LLVMTypes::Int32Type);
    } else if (maskStatus == MaskStatus::all_on) {
        // The mask is all on, so turn this into a regular store
        llvm::Instruction *store = nullptr;

        lvalue = new llvm::BitCastInst(lvalue, LLVMTypes::PtrType, "lvalue_to_ptr_type",
                                       ISPC_INSERTION_POINT_INSTRUCTION(callInst));
        LLVMCopyMetadata(lvalue, callInst);
        store = new llvm::StoreInst(
            rvalue, lvalue, false /* not volatile */,
            llvm::MaybeAlign(g->opt.forceAlignedMemory ? g->target->getNativeVectorAlignment() : align).valueOrOne());

        if (store != nullptr) {
            LLVMCopyMetadata(store, callInst);
            llvm::ReplaceInstWithInst(callInst, store);
            return store;
        }

#ifdef ISPC_XE_ENABLED
    } else {
        if (g->target->isXeTarget() && GetAddressSpace(lvalue) == AddressSpace::ispc_global) {
            // In this case we use masked_store which on Xe target causes scatter usage.
            // Get the source position from the metadata attached to the call
            // instruction so that we can issue PerformanceWarning()s below.
            SourcePos pos;
            bool gotPosition = LLVMGetSourcePosFromMetadata(callInst, &pos);
            if (gotPosition) {
                PerformanceWarning(pos, "Scatter required to store value.");
            }
        }
#endif
    }
    return nullptr;
}

static llvm::Value *lImproveMaskedLoad(const MemoryAccess &access, llvm::BasicBlock::iterator iter) {
    // A masked load of any flavor; it is assumed aligned to its element.
    if (access.kind() != MemoryOpKind::MaskedLoad) {
        return nullptr;
    }
    llvm::CallInst *callInst = access.call;
    int align = access.naturalAlignment();

    // Got one; grab the operands
    llvm::Value *ptr = access.base();
    llvm::Value *mask = access.mask();

    MaskStatus maskStatus = GetMaskStatusFromValue(mask);
    if (maskStatus == MaskStatus::all_off) {
        // Zero mask - no-op, so replace the load with a poison value
        llvm::Value *undef = llvm::PoisonValue::get(callInst->getType());
        ReplaceInstWithValueWrapper(iter, undef);
        return undef;
    } else if (maskStatus == MaskStatus::all_on) {
        // The mask is all on, so turn this into a regular load
        llvm::Instruction *load = nullptr;
        Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
        load = new llvm::LoadInst(
            callInst->getType(), ptr, callInst->getName(), false /* not volatile */,
            llvm::MaybeAlign(g->opt.forceAlignedMemory ? g->target->getNativeVectorAlignment() : align).valueOrOne());

        if (load != nullptr) {
            LLVMCopyMetadata(load, callInst);
            llvm::ReplaceInstWithInst(callInst, load);
            return load;
        }
    }
    return nullptr;
}

bool ImproveMemoryOpsPass::improveMemoryOps(llvm::BasicBlock &bb) {
    DEBUG_START_BB("ImproveMemoryOps");

    bool modifiedAny = false;

    // Iterate through all of the instructions in the basic block.
    // Note: we do modify instruction list during the traversal, so the iterator
    // is moved forward before the instruction is processed.
    for (llvm::BasicBlock::iterator iter = bb.begin(), e = bb.end(); iter != e;) {
        llvm::BasicBlock::iterator curIter = iter++;
        llvm::CallInst *callInst = llvm::dyn_cast<llvm::CallInst>(&*curIter);

        while (callInst != nullptr) {
            // Decode the call once; every optimization below reads its
            // operands by role. Calls that are not memory operations, or
            // whose signature does not match their name, are left alone.
            std::optional<MemoryAccess> access = DecodeMemoryAccess(callInst);
            if (!access) {
                break;
            }
            llvm::Value *newValue = nullptr;

            if ((newValue = lGSToGSBaseOffsets(*access))) {
                modifiedAny = true;
            } else if ((newValue = lGSBaseOffsetsGetMoreConst(*access))) {
                modifiedAny = true;
            } else if ((newValue = lGSToLoadStore(*access))) {
                modifiedAny = true;
            } else if ((newValue = lImproveMaskedStore(*access))) {
                modifiedAny = true;
            } else if ((newValue = lImproveMaskedLoad(*access, curIter))) {
                modifiedAny = true;
            }

            // More than one of optimizations above may be applied subsequently
            // to a single call. If the returned value is another call instruction,
            // try again.
            callInst = llvm::dyn_cast_or_null<llvm::CallInst>(newValue);
            // After the an old call instruction is removed, "curIter" is invalid,
            // so we reestablish it be moving back from "iter", which looks one
            // instruction ahead.
            curIter = iter;
            curIter--;
        }
    }

    DEBUG_END_BB("ImproveMemoryOps");

    return modifiedAny;
}

llvm::PreservedAnalyses ImproveMemoryOpsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::TimeTraceScope FuncScope("ImproveMemoryOpsPass::run", F.getName());
    bool modifiedAny = false;

    for (llvm::BasicBlock &BB : F) {
        modifiedAny |= improveMemoryOps(BB);
    }
    if (!modifiedAny) {
        // No changes, all analyses are preserved.
        return llvm::PreservedAnalyses::all();
    }

    llvm::PreservedAnalyses PA;
    PA.preserveSet<llvm::CFGAnalyses>();
    return PA;
}

} // namespace ispc
