/*
  Copyright (c) 2022-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

#include "ImproveMemoryOps.h"
#include "builtins-decl.h"

namespace ispc {

using namespace builtin;

/** Check to make sure that this value is actually a pointer in the end.
    We need to make sure that given an expression like vec(offset) +
    ptr2int(ptr), lGetBasePointer() doesn't return vec(offset) for the base
    pointer such that we then treat ptr2int(ptr) as an offset.  This ends
    up being important so that we don't generate LLVM GEP instructions like
    "gep inttoptr 8, i64 %ptr", which in turn can lead to incorrect code
    since LLVM's pointer aliasing analysis assumes that operands after the
    first one to a GEP aren't pointers.
 */
static llvm::Value *lCheckForActualPointer(llvm::Value *v) {
    if (v == nullptr) {
        return nullptr;
    } else if (llvm::isa<llvm::PointerType>(v->getType())) {
        return v;
    } else if (llvm::isa<llvm::PtrToIntInst>(v)) {
        return v;
    }
    // This one is tricky, as it's heuristic tuned for LLVM 3.7+, which may
    // optimize loading double* with consequent ptr2int to straight load of i64.
    // This heuristic should be good enough to catch all the cases we should
    // detect and nothing else.
    else if (llvm::isa<llvm::LoadInst>(v)) {
        return v;
    }

    else if (llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v)) {
        llvm::Value *t = lCheckForActualPointer(ci->getOperand(0));
        if (t == nullptr) {
            return nullptr;
        } else {
            return v;
        }
    } else {
        llvm::ConstantExpr *uce = llvm::dyn_cast<llvm::ConstantExpr>(v);
        if (uce != nullptr && uce->getOpcode() == llvm::Instruction::PtrToInt)
            return v;
        return nullptr;
    }
}

/** Given a llvm::Value representing a varying pointer, this function
    checks to see if all of the elements of the vector have the same value
    (i.e. there's a common base pointer). If broadcast has been already detected
    it checks that the first element of the vector is not undef. If one of the conditions
    is true, it returns the common pointer value; otherwise it returns nullptr.
 */
static llvm::Value *lGetBasePointer(llvm::Value *v, llvm::Instruction *insertBefore, bool broadcastDetected) {
    if (llvm::isa<llvm::InsertElementInst>(v) || llvm::isa<llvm::ShuffleVectorInst>(v)) {
        // If we have already detected broadcast we want to look for
        // the vector with the first not-undef element
        llvm::Value *element = LLVMFlattenInsertChain(v, g->target->getVectorWidth(), true, false, broadcastDetected);
        // TODO: it's probably ok to allow undefined elements and return
        // the base pointer if all of the other elements have the same
        // value.
        if (element != nullptr) {
            // all elements are the same and not nullptrs
            return lCheckForActualPointer(element);
        } else {
            return nullptr;
        }
    }

    // This case comes up with global/static arrays
    if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v)) {
        return lCheckForActualPointer(cv->getSplatValue());
    } else if (llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v)) {
        return lCheckForActualPointer(cdv->getSplatValue());
    }
    // It is a little bit tricky to use operations with pointers, casted to int with another bit size
    // but sometimes it is useful, so we handle this case here.
    else if (llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v)) {
        llvm::Value *t = lGetBasePointer(ci->getOperand(0), insertBefore, broadcastDetected);
        if (t == nullptr) {
            return nullptr;
        } else {
            return llvm::CastInst::Create(ci->getOpcode(), t, ci->getType()->getScalarType(),
                                          llvm::Twine(t->getName()) + "_cast", insertBefore);
        }
    }

    return nullptr;
}

/** Given the two operands to a constant add expression, see if we have the
    form "base pointer + offset", whee op0 is the base pointer and op1 is
    the offset; if so return the base and the offset. */
static llvm::Constant *lGetConstantAddExprBaseOffset(llvm::Constant *op0, llvm::Constant *op1, llvm::Constant **delta) {
    llvm::ConstantExpr *op = llvm::dyn_cast<llvm::ConstantExpr>(op0);
    if (op == nullptr || op->getOpcode() != llvm::Instruction::PtrToInt)
        // the first operand isn't a pointer
        return nullptr;

    llvm::ConstantInt *opDelta = llvm::dyn_cast<llvm::ConstantInt>(op1);
    if (opDelta == nullptr)
        // the second operand isn't an integer operand
        return nullptr;

    *delta = opDelta;
    return op0;
}

static llvm::Value *lExtractFromInserts(llvm::Value *v, unsigned int index) {
    llvm::InsertValueInst *iv = llvm::dyn_cast<llvm::InsertValueInst>(v);
    if (iv == nullptr)
        return nullptr;

    Assert(iv->hasIndices() && iv->getNumIndices() == 1);
    if (iv->getIndices()[0] == index)
        return iv->getInsertedValueOperand();
    else
        return lExtractFromInserts(iv->getAggregateOperand(), index);
}

/** Given a varying pointer in ptrs, this function checks to see if it can
    be determined to be indexing from a common uniform base pointer.  If
    so, the function returns the base pointer llvm::Value and initializes
    *offsets with an int vector of the per-lane offsets
 */
static llvm::Value *lGetBasePtrAndOffsets(llvm::Value *ptrs, llvm::Value **offsets, llvm::Instruction *insertBefore) {
    if (g->debugPrint) {
        fprintf(stderr, "lGetBasePtrAndOffsets\n");
        LLVMDumpValue(ptrs);
    }

    bool broadcastDetected = false;
    // Looking for %gep_offset = shufflevector <8 x i64> %0, <8 x i64> undef, <8 x i32> zeroinitializer
    llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(ptrs);
    if (shuffle != nullptr) {
        llvm::Value *indices = shuffle->getShuffleMaskForBitcode();
        llvm::Value *vec = shuffle->getOperand(1);

        if (LLVMIsValueUndef(vec) && llvm::isa<llvm::ConstantAggregateZero>(indices)) {
            broadcastDetected = true;
        }
    }
    llvm::Value *base = lGetBasePointer(ptrs, insertBefore, broadcastDetected);
    if (base != nullptr) {
        // We have a straight up varying pointer with no indexing that's
        // actually all the same value.
        if (g->target->is32Bit())
            *offsets = LLVMInt32Vector(0);
        else
            *offsets = LLVMInt64Vector((int64_t)0);

        if (broadcastDetected) {
            llvm::Value *op = shuffle->getOperand(0);
            llvm::BinaryOperator *bop_var = llvm::dyn_cast<llvm::BinaryOperator>(op);
            if (bop_var != nullptr &&
                ((bop_var->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop_var))) {
                // We expect here ConstantVector as
                // <i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>
                llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(bop_var->getOperand(1));
                llvm::Instruction *shuffle_offset = nullptr;
                if (cv != nullptr) {
                    llvm::Value *zeroMask = llvm::ConstantVector::getSplat(
                        llvm::ElementCount::get(llvm::dyn_cast<llvm::FixedVectorType>(cv->getType())->getNumElements(),
                                                false),
                        llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
                    // Create offset
                    shuffle_offset = new llvm::ShuffleVectorInst(cv, llvm::UndefValue::get(cv->getType()), zeroMask,
                                                                 "shuffle", bop_var);
                } else {
                    // or it binaryoperator can accept another binary operator
                    // that is a result of counting another part of offset:
                    // %another_bop = bop <16 x i32> %vec, <i32 7, i32 undef, i32 undef, ...>
                    // %offsets = add <16 x i32> %another_bop, %base
                    bop_var = llvm::dyn_cast<llvm::BinaryOperator>(bop_var->getOperand(0));
                    if (bop_var != nullptr) {
                        llvm::Type *bop_var_type = bop_var->getType();
                        llvm::Value *zeroMask = llvm::ConstantVector::getSplat(
                            llvm::ElementCount::get(
                                llvm::dyn_cast<llvm::FixedVectorType>(bop_var_type)->getNumElements(), false),
                            llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
                        shuffle_offset = new llvm::ShuffleVectorInst(bop_var, llvm::UndefValue::get(bop_var_type),
                                                                     zeroMask, "shuffle");
                        shuffle_offset->insertAfter(bop_var);
                    }
                }
                if (shuffle_offset != nullptr) {
                    *offsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets, shuffle_offset,
                                                            "new_offsets", insertBefore);
                    return base;
                } else {
                    // Base + offset pattern was not recognized
                    return nullptr;
                }
            }
        }
        return base;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(ptrs);
    if (bop != nullptr && ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop))) {
        // If we have a common pointer plus something, then we're also
        // good.
        if ((base = lGetBasePtrAndOffsets(bop->getOperand(0), offsets, insertBefore)) != nullptr) {
            *offsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets, bop->getOperand(1), "new_offsets",
                                                    insertBefore);
            return base;
        } else if ((base = lGetBasePtrAndOffsets(bop->getOperand(1), offsets, insertBefore)) != nullptr) {
            *offsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, *offsets, bop->getOperand(0), "new_offsets",
                                                    insertBefore);
            return base;
        }
    }
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(ptrs);
    if (cv != nullptr) {
        // Indexing into global arrays can lead to this form, with
        // ConstantVectors..
        llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
        for (int i = 0; i < (int)cv->getNumOperands(); ++i) {
            llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(cv->getOperand(i));
            if (c == nullptr)
                return nullptr;
            elements.push_back(c);
        }

        llvm::Constant *delta[ISPC_MAX_NVEC];
        for (unsigned int i = 0; i < elements.size(); ++i) {
            // For each element, try to decompose it into either a straight
            // up base pointer, or a base pointer plus an integer value.
            llvm::ConstantExpr *ce = llvm::dyn_cast<llvm::ConstantExpr>(elements[i]);
            if (ce == nullptr)
                return nullptr;

            delta[i] = nullptr;
            llvm::Value *elementBase = nullptr; // base pointer for this element
            if (ce->getOpcode() == llvm::Instruction::PtrToInt) {
                // If the element is just a ptr to int instruction, treat
                // it as having an offset of zero
                elementBase = ce;
                delta[i] = g->target->is32Bit() ? LLVMInt32(0) : LLVMInt64(0);
            } else if ((ce->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(ce)) {
                // Try both orderings of the operands to see if we can get
                // a pointer+offset out of them.
                elementBase = lGetConstantAddExprBaseOffset(ce->getOperand(0), ce->getOperand(1), &delta[i]);
                if (elementBase == nullptr)
                    elementBase = lGetConstantAddExprBaseOffset(ce->getOperand(1), ce->getOperand(0), &delta[i]);
            }

            // We weren't able to find a base pointer in the above.  (We
            // don't expect this to happen; if it does, it may be necessary
            // to handle more cases in the decomposition above.)
            if (elementBase == nullptr)
                return nullptr;

            Assert(delta[i] != nullptr);
            if (base == nullptr)
                // The first time we've found a base pointer
                base = elementBase;
            else if (base != elementBase)
                // Different program instances have different base
                // pointers, so no luck.
                return nullptr;
        }

        Assert(base != nullptr);
        llvm::ArrayRef<llvm::Constant *> deltas(&delta[0], &delta[elements.size()]);
        *offsets = llvm::ConstantVector::get(deltas);
        return base;
    }

    llvm::ExtractValueInst *ev = llvm::dyn_cast<llvm::ExtractValueInst>(ptrs);
    if (ev != nullptr) {
        Assert(ev->getNumIndices() == 1);
        int index = ev->getIndices()[0];
        ptrs = lExtractFromInserts(ev->getAggregateOperand(), index);
        if (ptrs != nullptr)
            return lGetBasePtrAndOffsets(ptrs, offsets, insertBefore);
    }

    return nullptr;
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
        // Check the cast target.
        llvm::Value *co, *vo;
        lExtractConstantOffset(cast->getOperand(0), &co, &vo, insertBefore);

        // make new cast instructions for the two parts
        if (co == nullptr)
            *constOffset = nullptr;
        else
            *constOffset = llvm::CastInst::Create(cast->getOpcode(), co, cast->getType(),
                                                  llvm::Twine(co->getName()) + "_cast", insertBefore);
        if (vo == nullptr)
            *variableOffset = nullptr;
        else
            *variableOffset = llvm::CastInst::Create(cast->getOpcode(), vo, cast->getType(),
                                                     llvm::Twine(vo->getName()) + "_cast", insertBefore);
        return;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(vec);
    if (bop != nullptr) {
        llvm::Value *op0 = bop->getOperand(0);
        llvm::Value *op1 = bop->getOperand(1);
        llvm::Value *c0, *v0, *c1, *v1;

        if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            if (c0 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(c0))
                *constOffset = c1;
            else if (c1 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(c1))
                *constOffset = c0;
            else
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, c0, c1, ((llvm::Twine("add_") + c0->getName()) + "_") + c1->getName(),
                    insertBefore);

            if (v0 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(v0))
                *variableOffset = v1;
            else if (v1 == nullptr || llvm::isa<llvm::ConstantAggregateZero>(v1))
                *variableOffset = v0;
            else
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, v0, v1, ((llvm::Twine("add_") + v0->getName()) + "_") + v1->getName(),
                    insertBefore);
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
                    insertBefore);
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Shl, v0, c1, ((llvm::Twine("shl_") + v0->getName()) + "_") + c1->getName(),
                    insertBefore);
            }
            return;
        } else if (bop->getOpcode() == llvm::Instruction::Mul) {
            lExtractConstantOffset(op0, &c0, &v0, insertBefore);
            lExtractConstantOffset(op1, &c1, &v1, insertBefore);

            // Given the product of constant and variable terms, we have:
            // (c0 + v0) * (c1 + v1) == (c0 c1) + (v0 c1 + c0 v1 + v0 v1)
            // Note that the first term is a constant and the last three are
            // variable.
            if (c0 != nullptr && c1 != nullptr)
                *constOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Mul, c0, c1, ((llvm::Twine("mul_") + c0->getName()) + "_") + c1->getName(),
                    insertBefore);
            else
                *constOffset = nullptr;

            llvm::Value *va = nullptr, *vb = nullptr, *vc = nullptr;
            if (v0 != nullptr && c1 != nullptr)
                va = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, c1,
                                                  ((llvm::Twine("mul_") + v0->getName()) + "_") + c1->getName(),
                                                  insertBefore);
            if (c0 != nullptr && v1 != nullptr)
                vb = llvm::BinaryOperator::Create(llvm::Instruction::Mul, c0, v1,
                                                  ((llvm::Twine("mul_") + c0->getName()) + "_") + v1->getName(),
                                                  insertBefore);
            if (v0 != nullptr && v1 != nullptr)
                vc = llvm::BinaryOperator::Create(llvm::Instruction::Mul, v0, v1,
                                                  ((llvm::Twine("mul_") + v0->getName()) + "_") + v1->getName(),
                                                  insertBefore);

            llvm::Value *vab = nullptr;
            if (va != nullptr && vb != nullptr)
                vab = llvm::BinaryOperator::Create(llvm::Instruction::Add, va, vb,
                                                   ((llvm::Twine("add_") + va->getName()) + "_") + vb->getName(),
                                                   insertBefore);
            else if (va != nullptr)
                vab = va;
            else
                vab = vb;

            if (vab != nullptr && vc != nullptr)
                *variableOffset = llvm::BinaryOperator::Create(
                    llvm::Instruction::Add, vab, vc, ((llvm::Twine("add_") + vab->getName()) + "_") + vc->getName(),
                    insertBefore);
            else if (vab != nullptr)
                *variableOffset = vab;
            else
                *variableOffset = vc;

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
    if (cvec == nullptr)
        return false;

    llvm::Constant *splatConst = cvec->getSplatValue();
    if (splatConst == nullptr)
        return false;

    llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(splatConst);
    if (ci == nullptr)
        return false;

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
            llvm::Value *splatDiv =
                llvm::BinaryOperator::Create(llvm::Instruction::SDiv, splatOperand, splatScaleVec, "div", insertBefore);
            *result = llvm::BinaryOperator::Create(llvm::Instruction::Mul, splatDiv, otherOperand, "mul", insertBefore);
            return LLVMInt32(scale);
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
        // Check the cast target.
        llvm::Value *scale = lExtractOffsetVector248Scale(&castOp);
        if (scale == nullptr)
            return nullptr;

        // make a new cast instruction so that we end up with the right
        // type
        *vec = llvm::CastInst::Create(cast->getOpcode(), castOp, cast->getType(), "offset_cast", cast);
        return scale;
    }

    // If we don't have a binary operator, then just give up
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(*vec);
    if (bop == nullptr)
        return LLVMInt32(1);

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
                *vec = llvm::BinaryOperator::Create(llvm::Instruction::Add, op0, op1, "new_add", bop);
                return s0;
            } else
                return LLVMInt32(1);
        }
    } else if (bop->getOpcode() == llvm::Instruction::Mul) {
        // Check each operand for being one of the scale factors we care about.
        int splat;
        if (lIsIntegerSplat(op0, &splat))
            return lExtract248Scale(op0, splat, op1, vec);
        else if (lIsIntegerSplat(op1, &splat))
            return lExtract248Scale(op1, splat, op0, vec);
        else
            return LLVMInt32(1);
    } else
        return LLVMInt32(1);
}

static bool lVectorIs32BitInts(llvm::Value *v) {
    int nElts;
    int64_t elts[ISPC_MAX_NVEC];
    if (!LLVMExtractVectorInts(v, elts, &nElts))
        return false;

    for (int i = 0; i < nElts; ++i)
        if ((int32_t)elts[i] != elts[i])
            return false;

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
        llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(variableOffset);
        if (sext != nullptr && sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType)
            // sext of a 32-bit vector -> the 32-bit vector is good
            variableOffset = sext->getOperand(0);
        else if (lVectorIs32BitInts(variableOffset))
            // The only constant vector we should have here is a vector of
            // all zeros (i.e. a ConstantAggregateZero, but just in case,
            // do the more general check with lVectorIs32BitInts().
            variableOffset = new llvm::TruncInst(variableOffset, LLVMTypes::Int32VectorType,
                                                 llvm::Twine(variableOffset->getName()) + "_trunc", insertBefore);
        else
            return false;
    }

    if (constOffset->getType() != LLVMTypes::Int32VectorType) {
        if (lVectorIs32BitInts(constOffset)) {
            // Truncate them so we have a 32-bit vector type for them.
            constOffset = new llvm::TruncInst(constOffset, LLVMTypes::Int32VectorType,
                                              llvm::Twine(constOffset->getName()) + "_trunc", insertBefore);
        } else {
            // FIXME: otherwise we just assume that all constant offsets
            // can actually always fit into 32-bits...  (This could be
            // wrong, but it should be only in pretty esoteric cases).  We
            // make this assumption for now since we sometimes generate
            // constants that need constant folding before we really have a
            // constant vector out of them, and
            // llvm::ConstantFoldInstruction() doesn't seem to be doing
            // enough for us in some cases if we call it from here.
            constOffset = new llvm::TruncInst(constOffset, LLVMTypes::Int32VectorType,
                                              llvm::Twine(constOffset->getName()) + "_trunc", insertBefore);
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
    // handle Adds, SExts, Constant Vectors
    if (llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v)) {
        if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
            return lIs32BitSafeHelper(bop->getOperand(0)) && lIs32BitSafeHelper(bop->getOperand(1));
        }
        return false;
    } else if (llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(v)) {
        return sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType;
    } else
        return lVectorIs32BitInts(v);
}

/** Check to see if the single offset vector can safely be represented with
    32-bit values.  If so, return true and update the pointed-to
    llvm::Value * to be the 32-bit equivalent. */
static bool lOffsets32BitSafe(llvm::Value **offsetPtr, llvm::Instruction *insertBefore) {
    llvm::Value *offset = *offsetPtr;

    if (offset->getType() == LLVMTypes::Int32VectorType)
        return true;

    llvm::SExtInst *sext = llvm::dyn_cast<llvm::SExtInst>(offset);
    if (sext != nullptr && sext->getOperand(0)->getType() == LLVMTypes::Int32VectorType) {
        // sext of a 32-bit vector -> the 32-bit vector is good
        *offsetPtr = sext->getOperand(0);
        return true;
    } else if (lIs32BitSafeHelper(offset)) {
        // The only constant vector we should have here is a vector of
        // all zeros (i.e. a ConstantAggregateZero, but just in case,
        // do the more general check with lVectorIs32BitInts().

        // Alternatively, offset could be a sequence of adds terminating
        // in safe constant vectors or a SExt.
        *offsetPtr = new llvm::TruncInst(offset, LLVMTypes::Int32VectorType, llvm::Twine(offset->getName()) + "_trunc",
                                         insertBefore);
        return true;
    } else
        return false;
}

// Contains a function call substitution info for gather/scatter/prefetch calls.
// We construct this class objects in program constructors phase, i.e., before initialization of g->target
// and any of LLVMTypes values. Moreover, g->target values are not same during multi-target compilation. Because of
// that, we postpone evaluation of types, choosing the proper function before actual call replacement here and in
// similar types across this file.
struct GSInfo {
    enum class Type {
        Gather,
        Scatter,
        Prefetch,
    };
    GSInfo(const char *pgbo64, const char *pgfbo64, const char *pgbo32, const char *pgfbo32, GSInfo::Type type)
        : pgbo64(pgbo64), pgfbo64(pgfbo64), pgbo32(pgbo32), pgfbo32(pgfbo32), type(type) {}
    GSInfo(const char *pgbo64, const char *pgfbo64, GSInfo::Type type)
        : pgbo64(pgbo64), pgfbo64(pgfbo64), pgbo32(pgbo64), pgfbo32(pgfbo64), type(type) {}
    static GSInfo Gather(const char *pgbo64, const char *pgfbo64, const char *pgbo32, const char *pgfbo32) {
        return GSInfo(pgbo64, pgfbo64, pgbo32, pgfbo32, Type::Gather);
    }
    static GSInfo Gather(const char *pgbo64, const char *pgfbo64) { return GSInfo(pgbo64, pgfbo64, Type::Gather); }
    static GSInfo Scatter(const char *pgbo64, const char *pgfbo64, const char *pgbo32, const char *pgfbo32) {
        return GSInfo(pgbo64, pgfbo64, pgbo32, pgfbo32, Type::Scatter);
    }
    static GSInfo Scatter(const char *pgbo64, const char *pgfbo64) { return GSInfo(pgbo64, pgfbo64, Type::Scatter); }
    static GSInfo Prefetch(const char *pgbo64, const char *pgfbo64) { return GSInfo(pgbo64, pgfbo64, Type::Prefetch); }
    bool isGather() const { return type == GSInfo::Type::Gather; }
    bool isScatter() const { return type == GSInfo::Type::Scatter; }
    bool isPrefetch() const { return type == GSInfo::Type::Prefetch; }
    llvm::Function *baseOffsets64Func(llvm::Module *M) const {
        const char *name = chooseBaseOrFactoredFunc() ? pgbo64 : pgfbo64;
        return M->getFunction(name);
    }
    llvm::Function *baseOffsets32Func(llvm::Module *M) const {
        const char *name = chooseBaseOrFactoredFunc() ? pgbo32 : pgfbo32;
        return M->getFunction(name);
    }

  private:
    bool chooseBaseOrFactoredFunc() const {
        switch (type) {
        case GSInfo::Type::Gather:
            return g->target->useGather();
        case GSInfo::Type::Scatter:
            return g->target->useScatter();
        case GSInfo::Type::Prefetch:
            return g->target->hasVecPrefetch();
        }
        return false;
    }
    const char *pgbo64;
    const char *pgfbo64;
    const char *pgbo32;
    const char *pgfbo32;
    GSInfo::Type type;
};

static llvm::CallInst *lGSToGSBaseOffsets(llvm::CallInst *callInst) {
    // A map for call replacement used in lGSToGSBaseOffsets.
    static std::unordered_map<std::string, GSInfo> replacementRules = {
        {__pseudo_gather32_i8,
         GSInfo::Gather(__pseudo_gather_base_offsets32_i8, __pseudo_gather_factored_base_offsets32_i8)},
        {__pseudo_gather32_i16,
         GSInfo::Gather(__pseudo_gather_base_offsets32_i16, __pseudo_gather_factored_base_offsets32_i16)},
        {__pseudo_gather32_half,
         GSInfo::Gather(__pseudo_gather_base_offsets32_half, __pseudo_gather_factored_base_offsets32_half)},
        {__pseudo_gather32_i32,
         GSInfo::Gather(__pseudo_gather_base_offsets32_i32, __pseudo_gather_factored_base_offsets32_i32)},
        {__pseudo_gather32_float,
         GSInfo::Gather(__pseudo_gather_base_offsets32_float, __pseudo_gather_factored_base_offsets32_float)},
        {__pseudo_gather32_i64,
         GSInfo::Gather(__pseudo_gather_base_offsets32_i64, __pseudo_gather_factored_base_offsets32_i64)},
        {__pseudo_gather32_double,
         GSInfo::Gather(__pseudo_gather_base_offsets32_double, __pseudo_gather_factored_base_offsets32_double)},
        {__pseudo_scatter32_i8,
         GSInfo::Scatter(__pseudo_scatter_base_offsets32_i8, __pseudo_scatter_factored_base_offsets32_i8)},
        {__pseudo_scatter32_i16,
         GSInfo::Scatter(__pseudo_scatter_base_offsets32_i16, __pseudo_scatter_factored_base_offsets32_i16)},
        {__pseudo_scatter32_half,
         GSInfo::Scatter(__pseudo_scatter_base_offsets32_half, __pseudo_scatter_factored_base_offsets32_half)},
        {__pseudo_scatter32_i32,
         GSInfo::Scatter(__pseudo_scatter_base_offsets32_i32, __pseudo_scatter_factored_base_offsets32_i32)},
        {__pseudo_scatter32_float,
         GSInfo::Scatter(__pseudo_scatter_base_offsets32_float, __pseudo_scatter_factored_base_offsets32_float)},
        {__pseudo_scatter32_i64,
         GSInfo::Scatter(__pseudo_scatter_base_offsets32_i64, __pseudo_scatter_factored_base_offsets32_i64)},
        {__pseudo_scatter32_double,
         GSInfo::Scatter(__pseudo_scatter_base_offsets32_double, __pseudo_scatter_factored_base_offsets32_double)},
        {__pseudo_gather64_i8,
         GSInfo::Gather(__pseudo_gather_base_offsets64_i8, __pseudo_gather_factored_base_offsets64_i8,
                        __pseudo_gather_base_offsets32_i8, __pseudo_gather_factored_base_offsets32_i8)},
        {__pseudo_gather64_i16,
         GSInfo::Gather(__pseudo_gather_base_offsets64_i16, __pseudo_gather_factored_base_offsets64_i16,
                        __pseudo_gather_base_offsets32_i16, __pseudo_gather_factored_base_offsets32_i16)},
        {__pseudo_gather64_half,
         GSInfo::Gather(__pseudo_gather_base_offsets64_half, __pseudo_gather_factored_base_offsets64_half,
                        __pseudo_gather_base_offsets32_half, __pseudo_gather_factored_base_offsets32_half)},
        {__pseudo_gather64_i32,
         GSInfo::Gather(__pseudo_gather_base_offsets64_i32, __pseudo_gather_factored_base_offsets64_i32,
                        __pseudo_gather_base_offsets32_i32, __pseudo_gather_factored_base_offsets32_i32)},
        {__pseudo_gather64_float,
         GSInfo::Gather(__pseudo_gather_base_offsets64_float, __pseudo_gather_factored_base_offsets64_float,
                        __pseudo_gather_base_offsets32_float, __pseudo_gather_factored_base_offsets32_float)},
        {__pseudo_gather64_i64,
         GSInfo::Gather(__pseudo_gather_base_offsets64_i64, __pseudo_gather_factored_base_offsets64_i64,
                        __pseudo_gather_base_offsets32_i64, __pseudo_gather_factored_base_offsets32_i64)},
        {__pseudo_gather64_double,
         GSInfo::Gather(__pseudo_gather_base_offsets64_double, __pseudo_gather_factored_base_offsets64_double,
                        __pseudo_gather_base_offsets32_double, __pseudo_gather_factored_base_offsets32_double)},
        {__pseudo_scatter64_i8,
         GSInfo::Scatter(__pseudo_scatter_base_offsets64_i8, __pseudo_scatter_factored_base_offsets64_i8,
                         __pseudo_scatter_base_offsets32_i8, __pseudo_scatter_factored_base_offsets32_i8)},
        {__pseudo_scatter64_i16,
         GSInfo::Scatter(__pseudo_scatter_base_offsets64_i16, __pseudo_scatter_factored_base_offsets64_i16,
                         __pseudo_scatter_base_offsets32_i16, __pseudo_scatter_factored_base_offsets32_i16)},
        {__pseudo_scatter64_half,
         GSInfo::Scatter(__pseudo_scatter_base_offsets64_half, __pseudo_scatter_factored_base_offsets64_half,
                         __pseudo_scatter_base_offsets32_half, __pseudo_scatter_factored_base_offsets32_half)},
        {__pseudo_scatter64_i32,
         GSInfo::Scatter(__pseudo_scatter_base_offsets64_i32, __pseudo_scatter_factored_base_offsets64_i32,
                         __pseudo_scatter_base_offsets32_i32, __pseudo_scatter_factored_base_offsets32_i32)},
        {__pseudo_scatter64_float,
         GSInfo::Scatter(__pseudo_scatter_base_offsets64_float, __pseudo_scatter_factored_base_offsets64_float,
                         __pseudo_scatter_base_offsets32_float, __pseudo_scatter_factored_base_offsets32_float)},
        {__pseudo_scatter64_i64,
         GSInfo::Scatter(__pseudo_scatter_base_offsets64_i64, __pseudo_scatter_factored_base_offsets64_i64,
                         __pseudo_scatter_base_offsets32_i64, __pseudo_scatter_factored_base_offsets32_i64)},
        {__pseudo_scatter64_double,
         GSInfo::Scatter(__pseudo_scatter_base_offsets64_double, __pseudo_scatter_factored_base_offsets64_double,
                         __pseudo_scatter_base_offsets32_double, __pseudo_scatter_factored_base_offsets32_double)},
        {__pseudo_prefetch_read_varying_1,
         GSInfo::Prefetch(__pseudo_prefetch_read_varying_1_native, __prefetch_read_varying_1)},
        {__pseudo_prefetch_read_varying_2,
         GSInfo::Prefetch(__pseudo_prefetch_read_varying_2_native, __prefetch_read_varying_2)},
        {__pseudo_prefetch_read_varying_3,
         GSInfo::Prefetch(__pseudo_prefetch_read_varying_3_native, __prefetch_read_varying_3)},
        {__pseudo_prefetch_read_varying_nt,
         GSInfo::Prefetch(__pseudo_prefetch_read_varying_nt_native, __prefetch_read_varying_nt)},
        {__pseudo_prefetch_write_varying_1,
         GSInfo::Prefetch(__pseudo_prefetch_write_varying_1_native, __prefetch_write_varying_1)},
        {__pseudo_prefetch_write_varying_2,
         GSInfo::Prefetch(__pseudo_prefetch_write_varying_2_native, __prefetch_write_varying_2)},
        {__pseudo_prefetch_write_varying_3,
         GSInfo::Prefetch(__pseudo_prefetch_write_varying_3_native, __prefetch_write_varying_3)},
    };

    auto name = callInst->getCalledFunction()->getName().str();
    auto it = replacementRules.find(name);
    if (it == replacementRules.end()) {
        // it is not a call of __pseudo function stored in replacementRules
        return nullptr;
    }
    GSInfo *info = &it->second;

    // Try to transform the array of pointers to a single base pointer
    // and an array of int32 offsets.  (All the hard work is done by
    // lGetBasePtrAndOffsets).
    llvm::Value *ptrs = callInst->getArgOperand(0);
    llvm::Value *offsetVector = nullptr;
    llvm::Value *basePtr = lGetBasePtrAndOffsets(ptrs, &offsetVector, callInst);

    if (basePtr == nullptr || offsetVector == nullptr || (info->isPrefetch() && !g->target->hasVecPrefetch())) {
        // It's actually a fully general gather/scatter with a varying
        // set of base pointers, so leave it as is and continune onward
        // to the next instruction...
        return nullptr;
    }
    // Cast the base pointer to a void *, since that's what the
    // __pseudo_*_base_offsets_* functions want.
    basePtr = new llvm::IntToPtrInst(basePtr, LLVMTypes::VoidPointerType, llvm::Twine(basePtr->getName()) + "_2void",
                                     callInst);
    LLVMCopyMetadata(basePtr, callInst);
    llvm::Module *M = callInst->getModule();
    llvm::Function *gatherScatterFunc = info->baseOffsets64Func(M);
    llvm::CallInst *newCall = nullptr;

    if ((info->isGather() && g->target->useGather()) || (info->isScatter() && g->target->useScatter()) ||
        (info->isPrefetch() && g->target->hasVecPrefetch())) {

        // See if the offsets are scaled by 2, 4, or 8.  If so,
        // extract that scale factor and rewrite the offsets to remove
        // it.
        llvm::Value *offsetScale = lExtractOffsetVector248Scale(&offsetVector);

        // If we're doing 32-bit addressing on a 64-bit target, here we
        // will see if we can call one of the 32-bit variants of the pseudo
        // gather/scatter functions.
        if (g->opt.force32BitAddressing && lOffsets32BitSafe(&offsetVector, callInst)) {
            gatherScatterFunc = info->baseOffsets32Func(M);
        }

        if (info->isGather() || info->isPrefetch()) {
            llvm::Value *mask = callInst->getArgOperand(1);

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
            llvm::Value *storeValue = callInst->getArgOperand(1);
            llvm::Value *mask = callInst->getArgOperand(2);

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
        if (constOffset == nullptr)
            constOffset = LLVMIntAsType(0, offsetVector->getType());
        if (variableOffset == nullptr)
            variableOffset = LLVMIntAsType(0, offsetVector->getType());

        // See if the varying component is scaled by 2, 4, or 8.  If so,
        // extract that scale factor and rewrite variableOffset to remove
        // it.  (This also is pulled out so that we can match the scales by
        // 2/4/8 offered by x86 addressing operators.)
        llvm::Value *offsetScale = lExtractOffsetVector248Scale(&variableOffset);

        // If we're doing 32-bit addressing on a 64-bit target, here we
        // will see if we can call one of the 32-bit variants of the pseudo
        // gather/scatter functions.
        if (g->opt.force32BitAddressing && lOffsets32BitSafe(&variableOffset, &constOffset, callInst)) {
            gatherScatterFunc = info->baseOffsets32Func(M);
        }

        if (info->isGather() || info->isPrefetch()) {
            llvm::Value *mask = callInst->getArgOperand(1);

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
            llvm::Value *storeValue = callInst->getArgOperand(1);
            llvm::Value *mask = callInst->getArgOperand(2);

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

/** Try to improve the decomposition between compile-time constant and
    compile-time unknown offsets in calls to the __pseudo_*_base_offsets*
    functions. After other optimizations have run, we will sometimes be
    able to pull more terms out of the unknown part and add them into the
    compile-time-known part.
 */
static llvm::CallInst *lGSBaseOffsetsGetMoreConst(llvm::CallInst *callInst) {
    static std::unordered_map<std::string, int> checksDecompositionOfArgs = {
        {__pseudo_gather_base_offsets64_i8, 1},
        {__pseudo_gather_base_offsets64_i16, 1},
        {__pseudo_gather_base_offsets64_half, 1},
        {__pseudo_gather_base_offsets64_i32, 1},
        {__pseudo_gather_base_offsets64_float, 1},
        {__pseudo_gather_base_offsets64_i64, 1},
        {__pseudo_gather_base_offsets64_double, 1},
        {__pseudo_gather_base_offsets32_i8, 1},
        {__pseudo_gather_base_offsets32_i16, 1},
        {__pseudo_gather_base_offsets32_half, 1},
        {__pseudo_gather_base_offsets32_i32, 1},
        {__pseudo_gather_base_offsets32_float, 1},
        {__pseudo_gather_base_offsets32_i64, 1},
        {__pseudo_gather_base_offsets32_double, 1},
        {__pseudo_gather_factored_base_offsets64_i8, 1},
        {__pseudo_gather_factored_base_offsets64_i16, 1},
        {__pseudo_gather_factored_base_offsets64_half, 1},
        {__pseudo_gather_factored_base_offsets64_i32, 1},
        {__pseudo_gather_factored_base_offsets64_float, 1},
        {__pseudo_gather_factored_base_offsets64_i64, 1},
        {__pseudo_gather_factored_base_offsets64_double, 1},
        {__pseudo_gather_factored_base_offsets32_i8, 1},
        {__pseudo_gather_factored_base_offsets32_i16, 1},
        {__pseudo_gather_factored_base_offsets32_half, 1},
        {__pseudo_gather_factored_base_offsets32_i32, 1},
        {__pseudo_gather_factored_base_offsets32_float, 1},
        {__pseudo_gather_factored_base_offsets32_i64, 1},
        {__pseudo_gather_factored_base_offsets32_double, 1},
        {__pseudo_scatter_base_offsets64_i8, 1},
        {__pseudo_scatter_base_offsets64_i16, 1},
        {__pseudo_scatter_base_offsets64_half, 1},
        {__pseudo_scatter_base_offsets64_i32, 1},
        {__pseudo_scatter_base_offsets64_float, 1},
        {__pseudo_scatter_base_offsets64_i64, 1},
        {__pseudo_scatter_base_offsets64_double, 1},
        {__pseudo_scatter_base_offsets32_i8, 1},
        {__pseudo_scatter_base_offsets32_i16, 1},
        {__pseudo_scatter_base_offsets32_half, 1},
        {__pseudo_scatter_base_offsets32_i32, 1},
        {__pseudo_scatter_base_offsets32_float, 1},
        {__pseudo_scatter_base_offsets32_i64, 1},
        {__pseudo_scatter_base_offsets32_double, 1},
        {__pseudo_scatter_factored_base_offsets64_i8, 1},
        {__pseudo_scatter_factored_base_offsets64_i16, 1},
        {__pseudo_scatter_factored_base_offsets64_half, 1},
        {__pseudo_scatter_factored_base_offsets64_i32, 1},
        {__pseudo_scatter_factored_base_offsets64_float, 1},
        {__pseudo_scatter_factored_base_offsets64_i64, 1},
        {__pseudo_scatter_factored_base_offsets64_double, 1},
        {__pseudo_scatter_factored_base_offsets32_i8, 1},
        {__pseudo_scatter_factored_base_offsets32_i16, 1},
        {__pseudo_scatter_factored_base_offsets32_half, 1},
        {__pseudo_scatter_factored_base_offsets32_i32, 1},
        {__pseudo_scatter_factored_base_offsets32_float, 1},
        {__pseudo_scatter_factored_base_offsets32_i64, 1},
        {__pseudo_scatter_factored_base_offsets32_double, 1},
        {__pseudo_prefetch_read_varying_1_native, 1},
        {__pseudo_prefetch_read_varying_2_native, 1},
        {__pseudo_prefetch_read_varying_3_native, 1},
        {__pseudo_prefetch_read_varying_nt_native, 1},
        {__prefetch_read_varying_1, 1},
        {__prefetch_read_varying_2, 1},
        {__prefetch_read_varying_3, 1},
        {__prefetch_read_varying_nt, 1},
        {__pseudo_prefetch_write_varying_1_native, 1},
        {__pseudo_prefetch_write_varying_2_native, 1},
        {__pseudo_prefetch_write_varying_3_native, 1},
        {__prefetch_write_varying_1, 1},
        {__prefetch_write_varying_2, 1},
        {__prefetch_write_varying_3, 1},
    };

    llvm::Function *calledFunc = callInst->getCalledFunction();
    Assert(calledFunc != nullptr);

    // Is one of the gather/scatter functins that decompose into
    // base+offsets being called?
    auto name = calledFunc->getName().str();
    auto it = checksDecompositionOfArgs.find(name);
    if (it == checksDecompositionOfArgs.end()) {
        // it is not a call of function stored in checksDecompositionOfArgs
        return nullptr;
    }

    // Grab the old variable offset
    llvm::Value *origVariableOffset = callInst->getArgOperand(1);

    // If it's zero, we're done.  Don't go and think that we're clever by
    // adding these zeros to the constant offsets.
    if (llvm::isa<llvm::ConstantAggregateZero>(origVariableOffset))
        return nullptr;

    // Try to decompose the old variable offset
    llvm::Value *constOffset = nullptr;
    llvm::Value *variableOffset = nullptr;
    lExtractConstantOffset(origVariableOffset, &constOffset, &variableOffset, callInst);

    // No luck
    if (constOffset == nullptr)
        return nullptr;

    // Total luck: everything could be moved to the constant offset
    if (variableOffset == nullptr)
        variableOffset = LLVMIntAsType(0, origVariableOffset->getType());

    // We need to scale the value we add to the constant offset by the
    // 2/4/8 scale for the variable offset, if present.
    llvm::ConstantInt *varScale = llvm::dyn_cast<llvm::ConstantInt>(callInst->getArgOperand(2));
    Assert(varScale != nullptr);

    llvm::Value *scaleSmear;
    if (origVariableOffset->getType() == LLVMTypes::Int64VectorType)
        scaleSmear = LLVMInt64Vector((int64_t)varScale->getZExtValue());
    else
        scaleSmear = LLVMInt32Vector((int32_t)varScale->getZExtValue());

    constOffset =
        llvm::BinaryOperator::Create(llvm::Instruction::Mul, constOffset, scaleSmear, constOffset->getName(), callInst);

    // And add the additional offset to the original constant offset
    constOffset = llvm::BinaryOperator::Create(llvm::Instruction::Add, constOffset, callInst->getArgOperand(3),
                                               callInst->getArgOperand(3)->getName(), callInst);

    // Finally, update the values of the operands to the gather/scatter
    // function.
    callInst->setArgOperand(1, variableOffset);
    callInst->setArgOperand(3, constOffset);

    return callInst;
}

static llvm::Value *lComputeCommonPointer(llvm::Value *base, llvm::Type *baseType, llvm::Value *offsets,
                                          llvm::Instruction *insertBefore) {
    llvm::Value *firstOffset = LLVMExtractFirstVectorElement(offsets);
    Assert(firstOffset != nullptr);

    return LLVMGEPInst(base, baseType, firstOffset, "ptr", insertBefore);
}

static llvm::Constant *lGetOffsetScaleVec(llvm::Value *offsetScale, llvm::Type *vecType) {
    llvm::ConstantInt *offsetScaleInt = llvm::dyn_cast<llvm::ConstantInt>(offsetScale);
    Assert(offsetScaleInt != nullptr);
    uint64_t scaleValue = offsetScaleInt->getZExtValue();

    std::vector<llvm::Constant *> scales;
    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
        if (vecType == LLVMTypes::Int64VectorType)
            scales.push_back(LLVMInt64(scaleValue));
        else {
            Assert(vecType == LLVMTypes::Int32VectorType);
            scales.push_back(LLVMInt32((int32_t)scaleValue));
        }
    }
    return llvm::ConstantVector::get(scales);
}

// alignment in bytes
enum class Alignment : int {
    A1 = 1,
    A2 = 2,
    A4 = 4,
    A8 = 8,
    A16 = 16,
};

struct GatherImpInfo {
    GatherImpInfo(const char *lmName, const char *bmName, llvm::Type **st, Alignment a)
        : load(lmName), blend(bmName), sType(st), alignment(a) {}
    bool isFactored() const { return !g->target->useGather(); }
    llvm::Function *loadMaskedFunc(llvm::Module *M) const { return M->getFunction(load); }
    llvm::Function *blendMaskedFunc(llvm::Module *M) const { return M->getFunction(blend); }
    llvm::Type *scalarType() const { return *sType; }
    llvm::Type *baseType() const {
        // Pseudo gather base pointer element type (the 1st argument of the intrinsic) is int8
        // e.g. @__pseudo_gather_base_offsets32_i8(i8 *, i32, <WIDTH x i32>, <WIDTH x MASK>)
        return LLVMTypes::Int8Type;
    }
    int align() const { return static_cast<int>(alignment); }

  private:
    const char *load;
    const char *blend;
    llvm::Type **sType;
    const Alignment alignment;
};

struct ScatterImpInfo {
    ScatterImpInfo(const char *msName, llvm::Type **vpt, Alignment a) : store(msName), vpType(vpt), alignment(a) {}
    bool isFactored() const { return !g->target->useScatter(); }
    llvm::Function *maskedStoreFunc(llvm::Module *M) const { return M->getFunction(store); }
    llvm::Type *vecPtrType() const { return *vpType; }
    llvm::Type *baseType() const {
        // Pseudo scatter base pointer element type (the 1st argument of the intrinsic) is int8
        // e.g. @__pseudo_scatter_base_offsets32_i8(i8 * nocapture, i32, <WIDTH x i32>, <WIDTH x i8>, <WIDTH x
        // MASK>)
        return LLVMTypes::Int8Type;
    }
    int align() const { return static_cast<int>(alignment); }

  private:
    const char *store;
    llvm::Type **vpType;
    const Alignment alignment;
};

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
static llvm::Instruction *lGSToLoadStore(llvm::CallInst *callInst) {

    static GatherImpInfo GII_i8 =
        GatherImpInfo(__masked_load_i8, __masked_load_blend_i8, &LLVMTypes::Int8Type, Alignment::A1);
    static GatherImpInfo GII_i16 =
        GatherImpInfo(__masked_load_i16, __masked_load_blend_i16, &LLVMTypes::Int16Type, Alignment::A2);
    static GatherImpInfo GII_half =
        GatherImpInfo(__masked_load_half, __masked_load_blend_half, &LLVMTypes::Float16Type, Alignment::A2);
    static GatherImpInfo GII_i32 =
        GatherImpInfo(__masked_load_i32, __masked_load_blend_i32, &LLVMTypes::Int32Type, Alignment::A4);
    static GatherImpInfo GII_float =
        GatherImpInfo(__masked_load_float, __masked_load_blend_float, &LLVMTypes::FloatType, Alignment::A4);
    static GatherImpInfo GII_i64 =
        GatherImpInfo(__masked_load_i64, __masked_load_blend_i64, &LLVMTypes::Int64Type, Alignment::A8);
    static GatherImpInfo GII_double =
        GatherImpInfo(__masked_load_double, __masked_load_blend_double, &LLVMTypes::DoubleType, Alignment::A8);

    static std::unordered_map<std::string, GatherImpInfo *> replGatherToLoad = {
        {__pseudo_gather_base_offsets32_i8, &GII_i8},
        {__pseudo_gather_factored_base_offsets32_i8, &GII_i8},
        {__pseudo_gather_base_offsets32_i16, &GII_i16},
        {__pseudo_gather_factored_base_offsets32_i16, &GII_i16},
        {__pseudo_gather_base_offsets32_half, &GII_half},
        {__pseudo_gather_factored_base_offsets32_half, &GII_half},
        {__pseudo_gather_base_offsets32_i32, &GII_i32},
        {__pseudo_gather_factored_base_offsets32_i32, &GII_i32},
        {__pseudo_gather_base_offsets32_float, &GII_float},
        {__pseudo_gather_factored_base_offsets32_float, &GII_float},
        {__pseudo_gather_base_offsets32_i64, &GII_i64},
        {__pseudo_gather_factored_base_offsets32_i64, &GII_i64},
        {__pseudo_gather_base_offsets32_double, &GII_double},
        {__pseudo_gather_factored_base_offsets32_double, &GII_double},
        {__pseudo_gather_base_offsets64_i8, &GII_i8},
        {__pseudo_gather_factored_base_offsets64_i8, &GII_i8},
        {__pseudo_gather_base_offsets64_i16, &GII_i16},
        {__pseudo_gather_factored_base_offsets64_i16, &GII_i16},
        {__pseudo_gather_base_offsets64_half, &GII_half},
        {__pseudo_gather_factored_base_offsets64_half, &GII_half},
        {__pseudo_gather_base_offsets64_i32, &GII_i32},
        {__pseudo_gather_factored_base_offsets64_i32, &GII_i32},
        {__pseudo_gather_base_offsets64_float, &GII_float},
        {__pseudo_gather_factored_base_offsets64_float, &GII_float},
        {__pseudo_gather_base_offsets64_i64, &GII_i64},
        {__pseudo_gather_factored_base_offsets64_i64, &GII_i64},
        {__pseudo_gather_base_offsets64_double, &GII_double},
        {__pseudo_gather_factored_base_offsets64_double, &GII_double},
    };

    static ScatterImpInfo SII_i8 =
        ScatterImpInfo(__pseudo_masked_store_i8, &LLVMTypes::Int8VectorPointerType, Alignment::A1);
    static ScatterImpInfo SII_i16 =
        ScatterImpInfo(__pseudo_masked_store_i16, &LLVMTypes::Int16VectorPointerType, Alignment::A2);
    static ScatterImpInfo SII_half =
        ScatterImpInfo(__pseudo_masked_store_half, &LLVMTypes::Float16VectorPointerType, Alignment::A2);
    static ScatterImpInfo SII_i32 =
        ScatterImpInfo(__pseudo_masked_store_i32, &LLVMTypes::Int32VectorPointerType, Alignment::A4);
    static ScatterImpInfo SII_float =
        ScatterImpInfo(__pseudo_masked_store_float, &LLVMTypes::FloatVectorPointerType, Alignment::A4);
    static ScatterImpInfo SII_i64 =
        ScatterImpInfo(__pseudo_masked_store_i64, &LLVMTypes::Int64VectorPointerType, Alignment::A8);
    static ScatterImpInfo SII_double =
        ScatterImpInfo(__pseudo_masked_store_double, &LLVMTypes::DoubleVectorPointerType, Alignment::A8);

    static std::unordered_map<std::string, ScatterImpInfo *> replScatterToStore = {
        {__pseudo_scatter_base_offsets32_i8, &SII_i8},
        {__pseudo_scatter_factored_base_offsets32_i8, &SII_i8},
        {__pseudo_scatter_base_offsets32_i16, &SII_i16},
        {__pseudo_scatter_factored_base_offsets32_i16, &SII_i16},
        {__pseudo_scatter_base_offsets32_half, &SII_half},
        {__pseudo_scatter_factored_base_offsets32_half, &SII_half},
        {__pseudo_scatter_base_offsets32_i32, &SII_i32},
        {__pseudo_scatter_factored_base_offsets32_i32, &SII_i32},
        {__pseudo_scatter_base_offsets32_float, &SII_float},
        {__pseudo_scatter_factored_base_offsets32_float, &SII_float},
        {__pseudo_scatter_base_offsets32_i64, &SII_i64},
        {__pseudo_scatter_factored_base_offsets32_i64, &SII_i64},
        {__pseudo_scatter_base_offsets32_double, &SII_double},
        {__pseudo_scatter_factored_base_offsets32_double, &SII_double},
        {__pseudo_scatter_base_offsets64_i8, &SII_i8},
        {__pseudo_scatter_factored_base_offsets64_i8, &SII_i8},
        {__pseudo_scatter_base_offsets64_i16, &SII_i16},
        {__pseudo_scatter_factored_base_offsets64_i16, &SII_i16},
        {__pseudo_scatter_base_offsets64_half, &SII_half},
        {__pseudo_scatter_factored_base_offsets64_half, &SII_half},
        {__pseudo_scatter_base_offsets64_i32, &SII_i32},
        {__pseudo_scatter_factored_base_offsets64_i32, &SII_i32},
        {__pseudo_scatter_base_offsets64_float, &SII_float},
        {__pseudo_scatter_factored_base_offsets64_float, &SII_float},
        {__pseudo_scatter_base_offsets64_i64, &SII_i64},
        {__pseudo_scatter_factored_base_offsets64_i64, &SII_i64},
        {__pseudo_scatter_base_offsets64_double, &SII_double},
        {__pseudo_scatter_factored_base_offsets64_double, &SII_double},
    };

    GatherImpInfo *gatherInfo = nullptr;
    ScatterImpInfo *scatterInfo = nullptr;
    llvm::Function *calledFunc = callInst->getCalledFunction();
    auto name = calledFunc->getName().str();

    auto itG = replGatherToLoad.find(name);
    if (itG != replGatherToLoad.end()) {
        gatherInfo = itG->second;
    }

    auto itS = replScatterToStore.find(name);
    if (itS != replScatterToStore.end()) {
        scatterInfo = itS->second;
    }

    if (gatherInfo == nullptr && scatterInfo == nullptr)
        return nullptr;

    SourcePos pos;
    LLVMGetSourcePosFromMetadata(callInst, &pos);

    llvm::Value *base = callInst->getArgOperand(0);
    llvm::Value *fullOffsets = nullptr;
    llvm::Value *storeValue = nullptr;
    llvm::Value *mask = nullptr;
    if ((gatherInfo != nullptr && gatherInfo->isFactored()) || (scatterInfo != nullptr && scatterInfo->isFactored())) {
        llvm::Value *varyingOffsets = callInst->getArgOperand(1);
        llvm::Value *offsetScale = callInst->getArgOperand(2);
        llvm::Value *constOffsets = callInst->getArgOperand(3);
        if (scatterInfo)
            storeValue = callInst->getArgOperand(4);
        mask = callInst->getArgOperand((gatherInfo != nullptr) ? 4 : 5);

        // Compute the full offset vector: offsetScale * varyingOffsets + constOffsets
        llvm::Constant *offsetScaleVec = lGetOffsetScaleVec(offsetScale, varyingOffsets->getType());

        llvm::Value *scaledVarying = llvm::BinaryOperator::Create(llvm::Instruction::Mul, offsetScaleVec,
                                                                  varyingOffsets, "scaled_varying", callInst);
        fullOffsets = llvm::BinaryOperator::Create(llvm::Instruction::Add, scaledVarying, constOffsets,
                                                   "varying+const_offsets", callInst);
    } else {
        if (scatterInfo)
            storeValue = callInst->getArgOperand(3);
        mask = callInst->getArgOperand((gatherInfo != nullptr) ? 3 : 4);

        llvm::Value *offsetScale = callInst->getArgOperand(1);
        llvm::Value *offsets = callInst->getArgOperand(2);
        llvm::Value *offsetScaleVec = lGetOffsetScaleVec(offsetScale, offsets->getType());

        fullOffsets =
            llvm::BinaryOperator::Create(llvm::Instruction::Mul, offsetScaleVec, offsets, "scaled_offsets", callInst);
    }

    Debug(SourcePos(), "GSToLoadStore: %s.", fullOffsets->getName().str().c_str());
    llvm::Type *scalarType =
        (gatherInfo != nullptr) ? gatherInfo->scalarType() : scatterInfo->vecPtrType()->getScalarType();

    if (LLVMVectorValuesAllEqual(fullOffsets)) {
        // If all the offsets are equal, then compute the single
        // pointer they all represent based on the first one of them
        // (arbitrarily).
        if (gatherInfo != nullptr) {
            // A gather with everyone going to the same location is
            // handled as a scalar load and broadcast across the lanes.
            Debug(pos, "Transformed gather to scalar load and broadcast!");
            llvm::Value *ptr;
            ptr = lComputeCommonPointer(base, gatherInfo->baseType(), fullOffsets, callInst);
            ptr = new llvm::BitCastInst(ptr, llvm::PointerType::get(scalarType, 0), base->getName(), callInst);

            LLVMCopyMetadata(ptr, callInst);
            Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
            llvm::Value *scalarValue = new llvm::LoadInst(scalarType, ptr, callInst->getName(), callInst);

            // Generate the following sequence:
            //   %name123 = insertelement <4 x i32> undef, i32 %val, i32 0
            //   %name124 = shufflevector <4 x i32> %name123, <4 x i32> undef,
            //                                              <4 x i32> zeroinitializer
            llvm::Value *undef1Value = llvm::UndefValue::get(callInst->getType());
            llvm::Value *undef2Value = llvm::UndefValue::get(callInst->getType());
            llvm::Value *insertVec =
                llvm::InsertElementInst::Create(undef1Value, scalarValue, LLVMInt32(0), callInst->getName(), callInst);
            llvm::Value *zeroMask = llvm::ConstantVector::getSplat(
                llvm::ElementCount::get(llvm::dyn_cast<llvm::FixedVectorType>(callInst->getType())->getNumElements(),
                                        false),
                llvm::Constant::getNullValue(llvm::Type::getInt32Ty(*g->ctx)));
            llvm::Instruction *shufInst =
                new llvm::ShuffleVectorInst(insertVec, undef2Value, zeroMask, callInst->getName());

            LLVMCopyMetadata(shufInst, callInst);
            llvm::ReplaceInstWithInst(callInst, shufInst);
            return shufInst;
        } else {
            // A scatter with everyone going to the same location is
            // undefined (if there's more than one program instance in
            // the gang).  Issue a warning.
            if (g->target->getVectorWidth() > 1)
                Warning(pos, "Undefined behavior: all program instances are "
                             "writing to the same location!");

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
        int step = gatherInfo ? gatherInfo->align() : scatterInfo->align();
        if (step > 0 && LLVMVectorIsLinear(fullOffsets, step)) {
            // We have a linear sequence of memory locations being accessed
            // starting with the location given by the offset from
            // offsetElements[0], with stride of 4 or 8 bytes (for 32 bit
            // and 64 bit gather/scatters, respectively.
            llvm::Value *ptr;
            llvm::Instruction *newCall = nullptr;
            llvm::Module *M = callInst->getModule();

            if (gatherInfo != nullptr) {
                ptr = lComputeCommonPointer(base, gatherInfo->baseType(), fullOffsets, callInst);
                LLVMCopyMetadata(ptr, callInst);
                Debug(pos, "Transformed gather to unaligned vector load!");
                bool doBlendLoad = false;
#ifdef ISPC_XE_ENABLED
                doBlendLoad = g->target->isXeTarget() && g->opt.enableXeUnsafeMaskedLoad;
#endif
                newCall = LLVMCallInst(doBlendLoad ? gatherInfo->blendMaskedFunc(M) : gatherInfo->loadMaskedFunc(M),
                                       ptr, mask, llvm::Twine(ptr->getName()) + "_masked_load");
            } else {
                Debug(pos, "Transformed scatter to unaligned vector store!");
                ptr = lComputeCommonPointer(base, scatterInfo->baseType(), fullOffsets, callInst);
                ptr = new llvm::BitCastInst(ptr, scatterInfo->vecPtrType(), "ptrcast", callInst);
                newCall = LLVMCallInst(scatterInfo->maskedStoreFunc(M), ptr, storeValue, mask, "");
            }
            LLVMCopyMetadata(newCall, callInst);
            llvm::ReplaceInstWithInst(callInst, newCall);
            return newCall;
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
static llvm::Value *lImproveMaskedStore(llvm::CallInst *callInst) {
    static std::unordered_map<std::string, Alignment> maskedStoreAlign = {
        {__pseudo_masked_store_i8, Alignment::A1},
        {__pseudo_masked_store_i16, Alignment::A2},
        {__pseudo_masked_store_half, Alignment::A2},
        {__pseudo_masked_store_i32, Alignment::A4},
        {__pseudo_masked_store_float, Alignment::A4},
        {__pseudo_masked_store_i64, Alignment::A8},
        {__pseudo_masked_store_double, Alignment::A8},
        {__masked_store_blend_i8, Alignment::A1},
        {__masked_store_blend_i16, Alignment::A2},
        {__masked_store_blend_half, Alignment::A2},
        {__masked_store_blend_i32, Alignment::A4},
        {__masked_store_blend_float, Alignment::A4},
        {__masked_store_blend_i64, Alignment::A8},
        {__masked_store_blend_double, Alignment::A8},
        {__masked_store_i8, Alignment::A1},
        {__masked_store_i16, Alignment::A2},
        {__masked_store_half, Alignment::A2},
        {__masked_store_i32, Alignment::A4},
        {__masked_store_float, Alignment::A4},
        {__masked_store_i64, Alignment::A8},
        {__masked_store_double, Alignment::A8},
    };

    llvm::Function *called = callInst->getCalledFunction();
    auto name = called->getName().str();
    auto it = maskedStoreAlign.find(name);
    if (it == maskedStoreAlign.end()) {
        // it is not a call of function stored in maskedStoreAlign
        return nullptr;
    }

    // Got one; grab the operands
    llvm::Value *lvalue = callInst->getArgOperand(0);
    llvm::Value *rvalue = callInst->getArgOperand(1);
    llvm::Value *mask = callInst->getArgOperand(2);
    int align = static_cast<int>(it->second);

    MaskStatus maskStatus = GetMaskStatusFromValue(mask);
    if (maskStatus == MaskStatus::all_off) {
        // Zero mask - no-op, so remove the store completely.  (This
        // may in turn lead to being able to optimize out instructions
        // that compute the rvalue...)
        callInst->eraseFromParent();
        // Return some fake undef value to signal that we did transformation.
        return llvm::UndefValue::get(LLVMTypes::Int32Type);
    } else if (maskStatus == MaskStatus::all_on) {
        // The mask is all on, so turn this into a regular store
        llvm::Type *rvalueType = rvalue->getType();
        llvm::Instruction *store = nullptr;
        llvm::Type *ptrType = llvm::PointerType::get(rvalueType, 0);

        lvalue = new llvm::BitCastInst(lvalue, ptrType, "lvalue_to_ptr_type", callInst);
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

static llvm::Value *lImproveMaskedLoad(llvm::CallInst *callInst, llvm::BasicBlock::iterator iter) {
    static std::unordered_map<std::string, Alignment> maskedLoadAlign = {
        {__masked_load_i8, Alignment::A1},        {__masked_load_i16, Alignment::A2},
        {__masked_load_half, Alignment::A2},      {__masked_load_i32, Alignment::A4},
        {__masked_load_float, Alignment::A4},     {__masked_load_i64, Alignment::A8},
        {__masked_load_double, Alignment::A8},    {__masked_load_blend_i8, Alignment::A1},
        {__masked_load_blend_i16, Alignment::A2}, {__masked_load_blend_half, Alignment::A2},
        {__masked_load_blend_i32, Alignment::A4}, {__masked_load_blend_float, Alignment::A4},
        {__masked_load_blend_i64, Alignment::A8}, {__masked_load_blend_double, Alignment::A8},
    };

    llvm::Function *called = callInst->getCalledFunction();
    auto name = called->getName().str();
    auto it = maskedLoadAlign.find(name);
    if (it == maskedLoadAlign.end()) {
        // it is not a call of function stored in maskedLoadAlign
        return nullptr;
    }
    int align = static_cast<int>(it->second);

    // Got one; grab the operands
    llvm::Value *ptr = callInst->getArgOperand(0);
    llvm::Value *mask = callInst->getArgOperand(1);

    MaskStatus maskStatus = GetMaskStatusFromValue(mask);
    if (maskStatus == MaskStatus::all_off) {
        // Zero mask - no-op, so replace the load with an undef value
        llvm::Value *undef = llvm::UndefValue::get(callInst->getType());
        ReplaceInstWithValueWrapper(iter, undef);
        return undef;
    } else if (maskStatus == MaskStatus::all_on) {
        // The mask is all on, so turn this into a regular load
        llvm::Instruction *load = nullptr;
        llvm::Type *ptrType = llvm::PointerType::get(callInst->getType(), 0);
        ptr = new llvm::BitCastInst(ptr, ptrType, "ptr_cast_for_load", callInst);
        Assert(llvm::isa<llvm::PointerType>(ptr->getType()));
        load = new llvm::LoadInst(
            callInst->getType(), ptr, callInst->getName(), false /* not volatile */,
            llvm::MaybeAlign(g->opt.forceAlignedMemory ? g->target->getNativeVectorAlignment() : align).valueOrOne(),
            (llvm::Instruction *)NULL);

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

        while (callInst && callInst->getCalledFunction()) {
            llvm::Value *newValue = nullptr;

            if ((newValue = lGSToGSBaseOffsets(callInst))) {
                modifiedAny = true;
            } else if ((newValue = lGSBaseOffsetsGetMoreConst(callInst))) {
                modifiedAny = true;
            } else if ((newValue = lGSToLoadStore(callInst))) {
                modifiedAny = true;
            } else if ((newValue = lImproveMaskedStore(callInst))) {
                modifiedAny = true;
            } else if ((newValue = lImproveMaskedLoad(callInst, curIter))) {
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

    // Skip __keep_funcs_live because it breaks some assumptions used here.
    // For example, lGSToLoadStore assumes that factored pseudo calls are
    // generated only when gathers are not used (useGather == false).
    if (F.getName().equals(__keep_funcs_live)) {
        return llvm::PreservedAnalyses::all();
    }

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
