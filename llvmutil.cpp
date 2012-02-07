/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

/** @file llvmutil.cpp
    @brief Implementations of various LLVM utility types and classes.
*/

#include "llvmutil.h"
#include "ispc.h"
#include "type.h"
#include <llvm/Instructions.h>

LLVM_TYPE_CONST llvm::Type *LLVMTypes::VoidType = NULL;
LLVM_TYPE_CONST llvm::PointerType *LLVMTypes::VoidPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::PointerIntType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::BoolType = NULL;

LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int8Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int16Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int32Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int64Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::FloatType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::DoubleType = NULL;

LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int8PointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int16PointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int32PointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int64PointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::FloatPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::DoublePointerType = NULL;

LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::MaskType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::BoolVectorType = NULL;

LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int1VectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int8VectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int16VectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int32VectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int64VectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::FloatVectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::DoubleVectorType = NULL;

LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int8VectorPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int16VectorPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int32VectorPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int64VectorPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::FloatVectorPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::DoubleVectorPointerType = NULL;

LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::VoidPointerVectorType = NULL;

llvm::Constant *LLVMTrue = NULL;
llvm::Constant *LLVMFalse = NULL;
llvm::Constant *LLVMMaskAllOn = NULL;
llvm::Constant *LLVMMaskAllOff = NULL;


void
InitLLVMUtil(llvm::LLVMContext *ctx, Target target) {
    LLVMTypes::VoidType = llvm::Type::getVoidTy(*ctx);
    LLVMTypes::VoidPointerType = llvm::PointerType::get(llvm::Type::getInt8Ty(*ctx), 0);
    LLVMTypes::PointerIntType = target.is32Bit ? llvm::Type::getInt32Ty(*ctx) :
        llvm::Type::getInt64Ty(*ctx);

    LLVMTypes::BoolType = llvm::Type::getInt1Ty(*ctx);
    LLVMTypes::Int8Type = llvm::Type::getInt8Ty(*ctx);
    LLVMTypes::Int16Type = llvm::Type::getInt16Ty(*ctx);
    LLVMTypes::Int32Type = llvm::Type::getInt32Ty(*ctx);
    LLVMTypes::Int64Type = llvm::Type::getInt64Ty(*ctx);
    LLVMTypes::FloatType = llvm::Type::getFloatTy(*ctx);
    LLVMTypes::DoubleType = llvm::Type::getDoubleTy(*ctx);

    LLVMTypes::Int8PointerType = llvm::PointerType::get(LLVMTypes::Int8Type, 0);
    LLVMTypes::Int16PointerType = llvm::PointerType::get(LLVMTypes::Int16Type, 0);
    LLVMTypes::Int32PointerType = llvm::PointerType::get(LLVMTypes::Int32Type, 0);
    LLVMTypes::Int64PointerType = llvm::PointerType::get(LLVMTypes::Int64Type, 0);
    LLVMTypes::FloatPointerType = llvm::PointerType::get(LLVMTypes::FloatType, 0);
    LLVMTypes::DoublePointerType = llvm::PointerType::get(LLVMTypes::DoubleType, 0);

    if (target.maskBitCount == 1)
        LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
            llvm::VectorType::get(llvm::Type::getInt1Ty(*ctx), target.vectorWidth);
    else {
        Assert(target.maskBitCount == 32);
        LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
            llvm::VectorType::get(llvm::Type::getInt32Ty(*ctx), target.vectorWidth);
    }

    LLVMTypes::Int1VectorType = 
        llvm::VectorType::get(llvm::Type::getInt1Ty(*ctx), target.vectorWidth);
    LLVMTypes::Int8VectorType = 
        llvm::VectorType::get(LLVMTypes::Int8Type, target.vectorWidth);
    LLVMTypes::Int16VectorType = 
        llvm::VectorType::get(LLVMTypes::Int16Type, target.vectorWidth);
    LLVMTypes::Int32VectorType = 
        llvm::VectorType::get(LLVMTypes::Int32Type, target.vectorWidth);
    LLVMTypes::Int64VectorType = 
        llvm::VectorType::get(LLVMTypes::Int64Type, target.vectorWidth);
    LLVMTypes::FloatVectorType = 
        llvm::VectorType::get(LLVMTypes::FloatType, target.vectorWidth);
    LLVMTypes::DoubleVectorType = 
        llvm::VectorType::get(LLVMTypes::DoubleType, target.vectorWidth);

    LLVMTypes::Int8VectorPointerType = llvm::PointerType::get(LLVMTypes::Int8VectorType, 0);
    LLVMTypes::Int16VectorPointerType = llvm::PointerType::get(LLVMTypes::Int16VectorType, 0);
    LLVMTypes::Int32VectorPointerType = llvm::PointerType::get(LLVMTypes::Int32VectorType, 0);
    LLVMTypes::Int64VectorPointerType = llvm::PointerType::get(LLVMTypes::Int64VectorType, 0);
    LLVMTypes::FloatVectorPointerType = llvm::PointerType::get(LLVMTypes::FloatVectorType, 0);
    LLVMTypes::DoubleVectorPointerType = llvm::PointerType::get(LLVMTypes::DoubleVectorType, 0);

    LLVMTypes::VoidPointerVectorType = g->target.is32Bit ? LLVMTypes::Int32VectorType :
        LLVMTypes::Int64VectorType;

    LLVMTrue = llvm::ConstantInt::getTrue(*ctx);
    LLVMFalse = llvm::ConstantInt::getFalse(*ctx);

    std::vector<llvm::Constant *> maskOnes;
    llvm::Constant *onMask = NULL;
    if (target.maskBitCount == 1)
        onMask = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*ctx), 1,
                                        false /*unsigned*/); // 0x1
    else
        onMask = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), -1,
                                    true /*signed*/); // 0xffffffff

    for (int i = 0; i < target.vectorWidth; ++i)
        maskOnes.push_back(onMask);
    LLVMMaskAllOn = llvm::ConstantVector::get(maskOnes);

    std::vector<llvm::Constant *> maskZeros;
    llvm::Constant *offMask = NULL;
    if (target.maskBitCount == 1)
        offMask = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*ctx), 0,
                                         true /*signed*/);
    else
        offMask = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), 0,
                                         true /*signed*/);

    for (int i = 0; i < target.vectorWidth; ++i)
        maskZeros.push_back(offMask);
    LLVMMaskAllOff = llvm::ConstantVector::get(maskZeros);
}


llvm::ConstantInt *
LLVMInt8(int8_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(*g->ctx), ival,
                                  true /*signed*/);
}


llvm::ConstantInt *
LLVMUInt8(uint8_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(*g->ctx), ival,
                                  false /*unsigned*/);
}


llvm::ConstantInt *
LLVMInt16(int16_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt16Ty(*g->ctx), ival,
                                  true /*signed*/);
}


llvm::ConstantInt *
LLVMUInt16(uint16_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt16Ty(*g->ctx), ival,
                                  false /*unsigned*/);
}


llvm::ConstantInt *
LLVMInt32(int32_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*g->ctx), ival,
                                  true /*signed*/);
}


llvm::ConstantInt *
LLVMUInt32(uint32_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*g->ctx), ival,
                                  false /*unsigned*/);
}


llvm::ConstantInt *
LLVMInt64(int64_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g->ctx), ival,
                                  true /*signed*/);
}


llvm::ConstantInt *
LLVMUInt64(uint64_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g->ctx), ival,
                                  false /*unsigned*/);
}


llvm::Constant *
LLVMFloat(float fval) {
    return llvm::ConstantFP::get(llvm::Type::getFloatTy(*g->ctx), fval);
}


llvm::Constant *
LLVMDouble(double dval) {
    return llvm::ConstantFP::get(llvm::Type::getDoubleTy(*g->ctx), dval);
}


llvm::Constant *
LLVMInt8Vector(int8_t ival) {
    llvm::Constant *v = LLVMInt8(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMInt8Vector(const int8_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMInt8(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt8Vector(uint8_t ival) {
    llvm::Constant *v = LLVMUInt8(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt8Vector(const uint8_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMUInt8(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMInt16Vector(int16_t ival) {
    llvm::Constant *v = LLVMInt16(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMInt16Vector(const int16_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMInt16(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt16Vector(uint16_t ival) {
    llvm::Constant *v = LLVMUInt16(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt16Vector(const uint16_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMUInt16(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMInt32Vector(int32_t ival) {
    llvm::Constant *v = LLVMInt32(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMInt32Vector(const int32_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMInt32(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt32Vector(uint32_t ival) {
    llvm::Constant *v = LLVMUInt32(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt32Vector(const uint32_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMUInt32(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMFloatVector(float fval) {
    llvm::Constant *v = LLVMFloat(fval);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMFloatVector(const float *fvec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMFloat(fvec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMDoubleVector(double dval) {
    llvm::Constant *v = LLVMDouble(dval);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMDoubleVector(const double *dvec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMDouble(dvec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMInt64Vector(int64_t ival) {
    llvm::Constant *v = LLVMInt64(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMInt64Vector(const int64_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMInt64(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt64Vector(uint64_t ival) {
    llvm::Constant *v = LLVMUInt64(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMUInt64Vector(const uint64_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(LLVMUInt64(ivec[i]));
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMBoolVector(bool b) {
    llvm::Constant *v;
    if (LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType) 
        v = llvm::ConstantInt::get(LLVMTypes::Int32Type, b ? 0xffffffff : 0, 
                                   false /*unsigned*/);
    else {
        Assert(LLVMTypes::BoolVectorType->getElementType() == 
               llvm::Type::getInt1Ty(*g->ctx));
        v = b ? LLVMTrue : LLVMFalse;
    }

    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}


llvm::Constant *
LLVMBoolVector(const bool *bvec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target.vectorWidth; ++i) {
        llvm::Constant *v;
        if (LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType) 
            v = llvm::ConstantInt::get(LLVMTypes::Int32Type, bvec[i] ? 0xffffffff : 0, 
                                       false /*unsigned*/);
        else {
            Assert(LLVMTypes::BoolVectorType->getElementType() == 
                   llvm::Type::getInt1Ty(*g->ctx));
            v = bvec[i] ? LLVMTrue : LLVMFalse;
        }

        vals.push_back(v);
    }
    return llvm::ConstantVector::get(vals);
}


/** Conservative test to see if two llvm::Values are equal.  There are
    (potentially many) cases where the two values actually are equal but
    this will return false.  However, if it does return true, the two
    vectors definitely are equal.

    @todo This seems to catch all of the cases we currently need it for in
    practice, but it's be nice to make it a little more robust/general.  In
    general, though, a little something called the halting problem means we
    won't get all of them.
*/
static bool
lValuesAreEqual(llvm::Value *v0, llvm::Value *v1, 
                std::vector<llvm::PHINode *> &seenPhi0,
                std::vector<llvm::PHINode *> &seenPhi1) {
    // Thanks to the fact that LLVM hashes and returns the same pointer for
    // constants (of all sorts, even constant expressions), this first test
    // actually catches a lot of cases.  LLVM's SSA form also helps a lot
    // with this..
    if (v0 == v1)
        return true;

    Assert(seenPhi0.size() == seenPhi1.size());
    for (unsigned int i = 0; i < seenPhi0.size(); ++i)
        if (v0 == seenPhi0[i] && v1 == seenPhi1[i])
            return true;

    llvm::BinaryOperator *bo0 = llvm::dyn_cast<llvm::BinaryOperator>(v0);
    llvm::BinaryOperator *bo1 = llvm::dyn_cast<llvm::BinaryOperator>(v1);
    if (bo0 != NULL && bo1 != NULL) {
        if (bo0->getOpcode() != bo1->getOpcode())
            return false;
        return (lValuesAreEqual(bo0->getOperand(0), bo1->getOperand(0),
                                seenPhi0, seenPhi1) &&
                lValuesAreEqual(bo0->getOperand(1), bo1->getOperand(1),
                                seenPhi0, seenPhi1));
    }

    llvm::PHINode *phi0 = llvm::dyn_cast<llvm::PHINode>(v0);
    llvm::PHINode *phi1 = llvm::dyn_cast<llvm::PHINode>(v1);
    if (phi0 != NULL && phi1 != NULL) {
        if (phi0->getNumIncomingValues() != phi1->getNumIncomingValues())
            return false;

        seenPhi0.push_back(phi0);
        seenPhi1.push_back(phi1);

        unsigned int numIncoming = phi0->getNumIncomingValues();
        // Check all of the incoming values: if all of them are all equal,
        // then we're good.
        bool anyFailure = false;
        for (unsigned int i = 0; i < numIncoming; ++i) {
            Assert(phi0->getIncomingBlock(i) == phi1->getIncomingBlock(i));
            if (!lValuesAreEqual(phi0->getIncomingValue(i), 
                                 phi1->getIncomingValue(i), seenPhi0, seenPhi1)) {
                anyFailure = true;
                break;
            }
        }

        seenPhi0.pop_back();
        seenPhi1.pop_back();

        return !anyFailure;
    }

    return false;
}


/** Given an llvm::Value known to be an integer, return its value as
    an int64_t.
*/
static int64_t
lGetIntValue(llvm::Value *offset) {
    llvm::ConstantInt *intOffset = llvm::dyn_cast<llvm::ConstantInt>(offset);
    Assert(intOffset && (intOffset->getBitWidth() == 32 ||
                         intOffset->getBitWidth() == 64));
    return intOffset->getSExtValue();
}


/** This function takes chains of InsertElement instructions along the
    lines of:

    %v0 = insertelement undef, value_0, i32 index_0
    %v1 = insertelement %v1,   value_1, i32 index_1
    ...
    %vn = insertelement %vn-1, value_n-1, i32 index_n-1

    and initializes the provided elements array such that the i'th
    llvm::Value * in the array is the element that was inserted into the
    i'th element of the vector.
*/
void
LLVMFlattenInsertChain(llvm::InsertElementInst *ie, int vectorWidth,
                       llvm::Value **elements) {
    for (int i = 0; i < vectorWidth; ++i)
        elements[i] = NULL;

    while (ie != NULL) {
        int64_t iOffset = lGetIntValue(ie->getOperand(2));
        Assert(iOffset >= 0 && iOffset < vectorWidth);
        Assert(elements[iOffset] == NULL);

        elements[iOffset] = ie->getOperand(1);

        llvm::Value *insertBase = ie->getOperand(0);
        ie = llvm::dyn_cast<llvm::InsertElementInst>(insertBase);
        if (ie == NULL) {
            if (llvm::isa<llvm::UndefValue>(insertBase))
                return;

            llvm::ConstantVector *cv = 
                llvm::dyn_cast<llvm::ConstantVector>(insertBase);
            Assert(cv != NULL);
            Assert(iOffset < (int)cv->getNumOperands());
            elements[iOffset] = cv->getOperand(iOffset);
        }
    }
}


/** Tests to see if all of the elements of the vector in the 'v' parameter
    are equal.  Like lValuesAreEqual(), this is a conservative test and may
    return false for arrays where the values are actually all equal.  */
bool
LLVMVectorValuesAllEqual(llvm::Value *v, int vectorLength,
                         std::vector<llvm::PHINode *> &seenPhis) {
    if (vectorLength == 1)
        return true;

    if (llvm::isa<llvm::ConstantAggregateZero>(v))
        return true;

    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v);
    if (cv != NULL)
        return (cv->getSplatValue() != NULL);

#ifdef LLVM_3_1svn
    llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v);
    if (cdv != NULL)
        return (cdv->getSplatValue() != NULL);
#endif

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v);
    if (bop != NULL)
        return (LLVMVectorValuesAllEqual(bop->getOperand(0), vectorLength, 
                                      seenPhis) &&
                LLVMVectorValuesAllEqual(bop->getOperand(1), vectorLength, 
                                      seenPhis));

    llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(v);
    if (cast != NULL)
        return LLVMVectorValuesAllEqual(cast->getOperand(0), vectorLength, 
                                     seenPhis);

    llvm::InsertElementInst *ie = llvm::dyn_cast<llvm::InsertElementInst>(v);
    if (ie != NULL) {
        llvm::Value *elements[ISPC_MAX_NVEC];
        LLVMFlattenInsertChain(ie, vectorLength, elements);

        // We will ignore any values of elements[] that are NULL; as they
        // correspond to undefined values--we just want to see if all of
        // the defined values have the same value.
        int lastNonNull = 0;
        while (lastNonNull < vectorLength && elements[lastNonNull] == NULL)
            ++lastNonNull;

        if (lastNonNull == vectorLength)
            // all of them are undef!
            return true;

        for (int i = lastNonNull; i < vectorLength; ++i) {
            if (elements[i] == NULL)
                continue;

            std::vector<llvm::PHINode *> seenPhi0;
            std::vector<llvm::PHINode *> seenPhi1;
            if (lValuesAreEqual(elements[lastNonNull], elements[i], seenPhi0, 
                                seenPhi1) == false)
                return false;
            lastNonNull = i;
        }
        return true;
    }

    llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(v);
    if (phi) {
        for (unsigned int i = 0; i < seenPhis.size(); ++i)
            if (seenPhis[i] == phi)
                return true;

        seenPhis.push_back(phi);

        unsigned int numIncoming = phi->getNumIncomingValues();
        // Check all of the incoming values: if all of them are all equal,
        // then we're good.
        for (unsigned int i = 0; i < numIncoming; ++i) {
            if (!LLVMVectorValuesAllEqual(phi->getIncomingValue(i), vectorLength,
                                       seenPhis)) {
                seenPhis.pop_back();
                return false;
            }
        }

        seenPhis.pop_back();
        return true;
    }

    if (llvm::isa<llvm::UndefValue>(v))
        // ?
        return false;

    Assert(!llvm::isa<llvm::Constant>(v));

    if (llvm::isa<llvm::CallInst>(v) || llvm::isa<llvm::LoadInst>(v) ||
        !llvm::isa<llvm::Instruction>(v))
        return false;

    llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(v);
    if (shuffle != NULL) {
        llvm::Value *indices = shuffle->getOperand(2);
        if (LLVMVectorValuesAllEqual(indices, vectorLength, seenPhis))
            // The easy case--just a smear of the same element across the
            // whole vector.
            return true;

        // TODO: handle more general cases?
        return false;
    }

#if 0
    fprintf(stderr, "all equal: ");
    v->dump();
    fprintf(stderr, "\n");
    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    if (inst) {
        inst->getParent()->dump();
        fprintf(stderr, "\n");
        fprintf(stderr, "\n");
    }
#endif

    return false;
}


