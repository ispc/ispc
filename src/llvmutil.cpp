/*
  Copyright (c) 2010-2021, Intel Corporation
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

#include <map>
#include <set>
#include <vector>

#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#ifdef ISPC_GENX_ENABLED
#include <llvm/GenXIntrinsics/GenXIntrinsics.h>
#endif

namespace ispc {

llvm::Type *LLVMTypes::VoidType = NULL;
llvm::PointerType *LLVMTypes::VoidPointerType = NULL;
llvm::Type *LLVMTypes::PointerIntType = NULL;
llvm::Type *LLVMTypes::BoolType = NULL;
llvm::Type *LLVMTypes::BoolStorageType = NULL;

llvm::Type *LLVMTypes::Int8Type = NULL;
llvm::Type *LLVMTypes::Int16Type = NULL;
llvm::Type *LLVMTypes::Int32Type = NULL;
llvm::Type *LLVMTypes::Int64Type = NULL;
llvm::Type *LLVMTypes::FloatType = NULL;
llvm::Type *LLVMTypes::DoubleType = NULL;

llvm::Type *LLVMTypes::Int8PointerType = NULL;
llvm::Type *LLVMTypes::Int16PointerType = NULL;
llvm::Type *LLVMTypes::Int32PointerType = NULL;
llvm::Type *LLVMTypes::Int64PointerType = NULL;
llvm::Type *LLVMTypes::FloatPointerType = NULL;
llvm::Type *LLVMTypes::DoublePointerType = NULL;

llvm::VectorType *LLVMTypes::MaskType = NULL;
llvm::VectorType *LLVMTypes::BoolVectorType = NULL;
llvm::VectorType *LLVMTypes::BoolVectorStorageType = NULL;

llvm::VectorType *LLVMTypes::Int1VectorType = NULL;
llvm::VectorType *LLVMTypes::Int8VectorType = NULL;
llvm::VectorType *LLVMTypes::Int16VectorType = NULL;
llvm::VectorType *LLVMTypes::Int32VectorType = NULL;
llvm::VectorType *LLVMTypes::Int64VectorType = NULL;
llvm::VectorType *LLVMTypes::FloatVectorType = NULL;
llvm::VectorType *LLVMTypes::DoubleVectorType = NULL;

llvm::Type *LLVMTypes::Int8VectorPointerType = NULL;
llvm::Type *LLVMTypes::Int16VectorPointerType = NULL;
llvm::Type *LLVMTypes::Int32VectorPointerType = NULL;
llvm::Type *LLVMTypes::Int64VectorPointerType = NULL;
llvm::Type *LLVMTypes::FloatVectorPointerType = NULL;
llvm::Type *LLVMTypes::DoubleVectorPointerType = NULL;

llvm::VectorType *LLVMTypes::VoidPointerVectorType = NULL;

llvm::Constant *LLVMTrue = NULL;
llvm::Constant *LLVMFalse = NULL;
llvm::Constant *LLVMTrueInStorage = NULL;
llvm::Constant *LLVMFalseInStorage = NULL;
llvm::Constant *LLVMMaskAllOn = NULL;
llvm::Constant *LLVMMaskAllOff = NULL;

void InitLLVMUtil(llvm::LLVMContext *ctx, Target &target) {
    LLVMTypes::VoidType = llvm::Type::getVoidTy(*ctx);
    LLVMTypes::VoidPointerType = llvm::PointerType::get(llvm::Type::getInt8Ty(*ctx), 0);
    LLVMTypes::PointerIntType = target.is32Bit() ? llvm::Type::getInt32Ty(*ctx) : llvm::Type::getInt64Ty(*ctx);

    LLVMTypes::BoolType = llvm::Type::getInt1Ty(*ctx);
    LLVMTypes::Int8Type = LLVMTypes::BoolStorageType = llvm::Type::getInt8Ty(*ctx);
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

    switch (target.getMaskBitCount()) {
    case 1:
        LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
            LLVMVECTOR::get(llvm::Type::getInt1Ty(*ctx), target.getVectorWidth());
        break;
    case 8:
        LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
            LLVMVECTOR::get(llvm::Type::getInt8Ty(*ctx), target.getVectorWidth());
        break;
    case 16:
        LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
            LLVMVECTOR::get(llvm::Type::getInt16Ty(*ctx), target.getVectorWidth());
        break;
    case 32:
        LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
            LLVMVECTOR::get(llvm::Type::getInt32Ty(*ctx), target.getVectorWidth());
        break;
    case 64:
        LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
            LLVMVECTOR::get(llvm::Type::getInt64Ty(*ctx), target.getVectorWidth());
        break;
    default:
        FATAL("Unhandled mask width for initializing MaskType");
    }

    LLVMTypes::Int1VectorType = LLVMVECTOR::get(llvm::Type::getInt1Ty(*ctx), target.getVectorWidth());
    LLVMTypes::Int8VectorType = LLVMTypes::BoolVectorStorageType =
        LLVMVECTOR::get(LLVMTypes::Int8Type, target.getVectorWidth());
    LLVMTypes::Int16VectorType = LLVMVECTOR::get(LLVMTypes::Int16Type, target.getVectorWidth());
    LLVMTypes::Int32VectorType = LLVMVECTOR::get(LLVMTypes::Int32Type, target.getVectorWidth());
    LLVMTypes::Int64VectorType = LLVMVECTOR::get(LLVMTypes::Int64Type, target.getVectorWidth());
    LLVMTypes::FloatVectorType = LLVMVECTOR::get(LLVMTypes::FloatType, target.getVectorWidth());
    LLVMTypes::DoubleVectorType = LLVMVECTOR::get(LLVMTypes::DoubleType, target.getVectorWidth());

    LLVMTypes::Int8VectorPointerType = llvm::PointerType::get(LLVMTypes::Int8VectorType, 0);
    LLVMTypes::Int16VectorPointerType = llvm::PointerType::get(LLVMTypes::Int16VectorType, 0);
    LLVMTypes::Int32VectorPointerType = llvm::PointerType::get(LLVMTypes::Int32VectorType, 0);
    LLVMTypes::Int64VectorPointerType = llvm::PointerType::get(LLVMTypes::Int64VectorType, 0);
    LLVMTypes::FloatVectorPointerType = llvm::PointerType::get(LLVMTypes::FloatVectorType, 0);
    LLVMTypes::DoubleVectorPointerType = llvm::PointerType::get(LLVMTypes::DoubleVectorType, 0);

    LLVMTypes::VoidPointerVectorType = g->target->is32Bit() ? LLVMTypes::Int32VectorType : LLVMTypes::Int64VectorType;

    LLVMTrue = llvm::ConstantInt::getTrue(*ctx);
    LLVMFalse = llvm::ConstantInt::getFalse(*ctx);
    LLVMTrueInStorage = llvm::ConstantInt::get(LLVMTypes::Int8Type, 0xff, false /*unsigned*/);
    LLVMFalseInStorage = llvm::ConstantInt::get(LLVMTypes::Int8Type, 0x00, false /*unsigned*/);

    std::vector<llvm::Constant *> maskOnes;
    llvm::Constant *onMask = NULL;
    switch (target.getMaskBitCount()) {
    case 1:
        onMask = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*ctx), 1, false /*unsigned*/); // 0x1
        break;
    case 8:
        onMask = llvm::ConstantInt::get(llvm::Type::getInt8Ty(*ctx), -1, true /*signed*/); // 0xff
        break;
    case 16:
        onMask = llvm::ConstantInt::get(llvm::Type::getInt16Ty(*ctx), -1, true /*signed*/); // 0xffff
        break;
    case 32:
        onMask = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), -1, true /*signed*/); // 0xffffffff
        break;
    case 64:
        onMask = llvm::ConstantInt::get(llvm::Type::getInt64Ty(*ctx), -1, true /*signed*/); // 0xffffffffffffffffull
        break;
    default:
        FATAL("Unhandled mask width for onMask");
    }

    for (int i = 0; i < target.getVectorWidth(); ++i)
        maskOnes.push_back(onMask);
    LLVMMaskAllOn = llvm::ConstantVector::get(maskOnes);

    std::vector<llvm::Constant *> maskZeros;
    llvm::Constant *offMask = NULL;
    switch (target.getMaskBitCount()) {
    case 1:
        offMask = llvm::ConstantInt::get(llvm::Type::getInt1Ty(*ctx), 0, true /*signed*/);
        break;
    case 8:
        offMask = llvm::ConstantInt::get(llvm::Type::getInt8Ty(*ctx), 0, true /*signed*/);
        break;
    case 16:
        offMask = llvm::ConstantInt::get(llvm::Type::getInt16Ty(*ctx), 0, true /*signed*/);
        break;
    case 32:
        offMask = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), 0, true /*signed*/);
        break;
    case 64:
        offMask = llvm::ConstantInt::get(llvm::Type::getInt64Ty(*ctx), 0, true /*signed*/);
        break;
    default:
        FATAL("Unhandled mask width for offMask");
    }
    for (int i = 0; i < target.getVectorWidth(); ++i)
        maskZeros.push_back(offMask);
    LLVMMaskAllOff = llvm::ConstantVector::get(maskZeros);
}

llvm::ConstantInt *LLVMInt8(int8_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(*g->ctx), ival, true /*signed*/);
}

llvm::ConstantInt *LLVMUInt8(uint8_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt8Ty(*g->ctx), ival, false /*unsigned*/);
}

llvm::ConstantInt *LLVMInt16(int16_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt16Ty(*g->ctx), ival, true /*signed*/);
}

llvm::ConstantInt *LLVMUInt16(uint16_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt16Ty(*g->ctx), ival, false /*unsigned*/);
}

llvm::ConstantInt *LLVMInt32(int32_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*g->ctx), ival, true /*signed*/);
}

llvm::ConstantInt *LLVMUInt32(uint32_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt32Ty(*g->ctx), ival, false /*unsigned*/);
}

llvm::ConstantInt *LLVMInt64(int64_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g->ctx), ival, true /*signed*/);
}

llvm::ConstantInt *LLVMUInt64(uint64_t ival) {
    return llvm::ConstantInt::get(llvm::Type::getInt64Ty(*g->ctx), ival, false /*unsigned*/);
}

llvm::Constant *LLVMFloat(float fval) { return llvm::ConstantFP::get(llvm::Type::getFloatTy(*g->ctx), fval); }

llvm::Constant *LLVMDouble(double dval) { return llvm::ConstantFP::get(llvm::Type::getDoubleTy(*g->ctx), dval); }

llvm::Constant *LLVMInt8Vector(int8_t ival) {
    llvm::Constant *v = LLVMInt8(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMInt8Vector(const int8_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMInt8(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt8Vector(uint8_t ival) {
    llvm::Constant *v = LLVMUInt8(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt8Vector(const uint8_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMUInt8(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMInt16Vector(int16_t ival) {
    llvm::Constant *v = LLVMInt16(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMInt16Vector(const int16_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMInt16(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt16Vector(uint16_t ival) {
    llvm::Constant *v = LLVMUInt16(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt16Vector(const uint16_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMUInt16(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMInt32Vector(int32_t ival) {
    llvm::Constant *v = LLVMInt32(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMInt32Vector(const int32_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMInt32(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt32Vector(uint32_t ival) {
    llvm::Constant *v = LLVMUInt32(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt32Vector(const uint32_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMUInt32(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMFloatVector(float fval) {
    llvm::Constant *v = LLVMFloat(fval);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMFloatVector(const float *fvec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMFloat(fvec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMDoubleVector(double dval) {
    llvm::Constant *v = LLVMDouble(dval);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMDoubleVector(const double *dvec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMDouble(dvec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMInt64Vector(int64_t ival) {
    llvm::Constant *v = LLVMInt64(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMInt64Vector(const int64_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMInt64(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt64Vector(uint64_t ival) {
    llvm::Constant *v = LLVMUInt64(ival);
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMUInt64Vector(const uint64_t *ivec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(LLVMUInt64(ivec[i]));
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMBoolVector(bool b) {
    llvm::Constant *v;
    if (LLVMTypes::BoolVectorType == LLVMTypes::Int64VectorType)
        v = llvm::ConstantInt::get(LLVMTypes::Int64Type, b ? 0xffffffffffffffffull : 0, false /*unsigned*/);
    else if (LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
        v = llvm::ConstantInt::get(LLVMTypes::Int32Type, b ? 0xffffffff : 0, false /*unsigned*/);
    else if (LLVMTypes::BoolVectorType == LLVMTypes::Int16VectorType)
        v = llvm::ConstantInt::get(LLVMTypes::Int16Type, b ? 0xffff : 0, false /*unsigned*/);
    else if (LLVMTypes::BoolVectorType == LLVMTypes::Int8VectorType)
        v = llvm::ConstantInt::get(LLVMTypes::Int8Type, b ? 0xff : 0, false /*unsigned*/);
    else {
        Assert(LLVMTypes::BoolVectorType == LLVMTypes::Int1VectorType);
        v = b ? LLVMTrue : LLVMFalse;
    }

    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMBoolVector(const bool *bvec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
        llvm::Constant *v;
        if (LLVMTypes::BoolVectorType == LLVMTypes::Int64VectorType)
            v = llvm::ConstantInt::get(LLVMTypes::Int64Type, bvec[i] ? 0xffffffffffffffffull : 0, false /*unsigned*/);
        else if (LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
            v = llvm::ConstantInt::get(LLVMTypes::Int32Type, bvec[i] ? 0xffffffff : 0, false /*unsigned*/);
        else if (LLVMTypes::BoolVectorType == LLVMTypes::Int16VectorType)
            v = llvm::ConstantInt::get(LLVMTypes::Int16Type, bvec[i] ? 0xffff : 0, false /*unsigned*/);
        else if (LLVMTypes::BoolVectorType == LLVMTypes::Int8VectorType)
            v = llvm::ConstantInt::get(LLVMTypes::Int8Type, bvec[i] ? 0xff : 0, false /*unsigned*/);
        else {
            Assert(LLVMTypes::BoolVectorType == LLVMTypes::Int1VectorType);
            v = bvec[i] ? LLVMTrue : LLVMFalse;
        }

        vals.push_back(v);
    }
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMBoolVectorInStorage(bool b) {
    llvm::Constant *v = b ? LLVMTrueInStorage : LLVMFalseInStorage;
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i)
        vals.push_back(v);
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMBoolVectorInStorage(const bool *bvec) {
    std::vector<llvm::Constant *> vals;
    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
        llvm::Constant *v = llvm::ConstantInt::get(LLVMTypes::Int8Type, bvec[i] ? 0xff : 0, false /*unsigned*/);
        vals.push_back(v);
    }
    return llvm::ConstantVector::get(vals);
}

llvm::Constant *LLVMIntAsType(int64_t val, llvm::Type *type) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vecType = llvm::dyn_cast<llvm::FixedVectorType>(type);
#else
    llvm::VectorType *vecType = llvm::dyn_cast<llvm::VectorType>(type);
#endif

    if (vecType != NULL) {
        llvm::Constant *v = llvm::ConstantInt::get(vecType->getElementType(), val, true /* signed */);
        std::vector<llvm::Constant *> vals;
        for (int i = 0; i < (int)vecType->getNumElements(); ++i)
            vals.push_back(v);
        return llvm::ConstantVector::get(vals);
    } else
        return llvm::ConstantInt::get(type, val, true /* signed */);
}

llvm::Constant *LLVMUIntAsType(uint64_t val, llvm::Type *type) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vecType = llvm::dyn_cast<llvm::FixedVectorType>(type);
#else
    llvm::VectorType *vecType = llvm::dyn_cast<llvm::VectorType>(type);
#endif

    if (vecType != NULL) {
        llvm::Constant *v = llvm::ConstantInt::get(vecType->getElementType(), val, false /* unsigned */);
        std::vector<llvm::Constant *> vals;
        for (int i = 0; i < (int)vecType->getNumElements(); ++i)
            vals.push_back(v);
        return llvm::ConstantVector::get(vals);
    } else
        return llvm::ConstantInt::get(type, val, false /* unsigned */);
}

/** Conservative test to see if two llvm::Values are equal.  There are
    (potentially many) cases where the two values actually are equal but
    this will return false.  However, if it does return true, the two
    vectors definitely are equal.
*/
static bool lValuesAreEqual(llvm::Value *v0, llvm::Value *v1, std::vector<llvm::PHINode *> &seenPhi0,
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
        return (lValuesAreEqual(bo0->getOperand(0), bo1->getOperand(0), seenPhi0, seenPhi1) &&
                lValuesAreEqual(bo0->getOperand(1), bo1->getOperand(1), seenPhi0, seenPhi1));
    }

    llvm::CastInst *cast0 = llvm::dyn_cast<llvm::CastInst>(v0);
    llvm::CastInst *cast1 = llvm::dyn_cast<llvm::CastInst>(v1);
    if (cast0 != NULL && cast1 != NULL) {
        if (cast0->getOpcode() != cast1->getOpcode())
            return false;
        return lValuesAreEqual(cast0->getOperand(0), cast1->getOperand(0), seenPhi0, seenPhi1);
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
            // FIXME: should it be ok if the incoming blocks are different,
            // where we just return faliure in this case?
            Assert(phi0->getIncomingBlock(i) == phi1->getIncomingBlock(i));
            if (!lValuesAreEqual(phi0->getIncomingValue(i), phi1->getIncomingValue(i), seenPhi0, seenPhi1)) {
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
static int64_t lGetIntValue(llvm::Value *offset) {
    llvm::ConstantInt *intOffset = llvm::dyn_cast<llvm::ConstantInt>(offset);
    Assert(intOffset && (intOffset->getBitWidth() == 32 || intOffset->getBitWidth() == 64));
    return intOffset->getSExtValue();
}

/**  Recognizes constant vector with undef operands except the first one:
 *   <i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>
 */
static bool lIsFirstElementConstVector(llvm::Value *v) {
    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v);
    // FIXME: skipping instruction and using getOperand(1) without checking for instruction type is incorrect!
    // This yields failing tests/write-same-loc.ispc for 32 bit targets (x86).
    // Need to understand what initial intent was here (what instruction supposed to be handled).
    // TODO: after fixing FIXME above isGenXTarget() needs to be removed.
    if (g->target->isGenXTarget()) {
        if (cv == NULL && llvm::isa<llvm::Instruction>(v)) {
            cv = llvm::dyn_cast<llvm::ConstantVector>(llvm::dyn_cast<llvm::Instruction>(v)->getOperand(1));
        }
    }
    if (cv != NULL) {
        llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(cv->getOperand(0));
        if (c == NULL) {
            return false;
        }

        for (int i = 1; i < (int)cv->getNumOperands(); ++i) {
            if (!llvm::isa<llvm::UndefValue>(cv->getOperand(i))) {
                return false;
            }
        }
        return true;
    }
    return false;
}

llvm::Value *LLVMFlattenInsertChain(llvm::Value *inst, int vectorWidth, bool compare, bool undef,
                                    bool searchFirstUndef) {
    std::vector<llvm::Value *> elements(vectorWidth, nullptr);

    // Catch a pattern of InsertElement chain.
    if (llvm::InsertElementInst *ie = llvm::dyn_cast<llvm::InsertElementInst>(inst)) {
        // Gather elements of vector
        while (ie != NULL) {
            int64_t iOffset = lGetIntValue(ie->getOperand(2));
            Assert(iOffset >= 0 && iOffset < vectorWidth);

            // Get the scalar value from this insert
            if (elements[iOffset] == NULL) {
                elements[iOffset] = ie->getOperand(1);
            }

            // Do we have another insert?
            llvm::Value *insertBase = ie->getOperand(0);
            ie = llvm::dyn_cast<llvm::InsertElementInst>(insertBase);
            if (ie != NULL) {
                continue;
            }

            if (llvm::isa<llvm::UndefValue>(insertBase)) {
                break;
            }

            if (llvm::isa<llvm::ConstantVector>(insertBase) || llvm::isa<llvm::ConstantAggregateZero>(insertBase)) {
                llvm::Constant *cv = llvm::dyn_cast<llvm::Constant>(insertBase);
                Assert(vectorWidth == (int)(cv->getNumOperands()));
                for (int i = 0; i < vectorWidth; i++) {
                    if (elements[i] == NULL) {
                        elements[i] = cv->getOperand(i);
                    }
                }
                break;
            } else {
                // Here chain ends in llvm::LoadInst or some other.
                // They are not equal to each other so we should return NULL if compare
                // and first element if we have it.
                Assert(compare == true || elements[0] != NULL);
                if (compare) {
                    return NULL;
                } else {
                    return elements[0];
                }
            }
            // TODO: Also, should we handle some other values like
            // ConstantDataVectors.
        }
        if (compare == false) {
            // We simply want first element
            return elements[0];
        }

        int null_number = 0;
        int NonNull = 0;
        for (int i = 0; i < vectorWidth; i++) {
            if (elements[i] == NULL) {
                null_number++;
            } else {
                NonNull = i;
            }
        }
        if (null_number == vectorWidth) {
            // All of elements are NULLs
            return NULL;
        }
        if ((undef == false) && (null_number != 0)) {
            // We don't want NULLs in chain, but we have them
            return NULL;
        }

        // Compare elements of vector
        for (int i = 0; i < vectorWidth; i++) {
            if (elements[i] == NULL) {
                continue;
            }

            std::vector<llvm::PHINode *> seenPhi0;
            std::vector<llvm::PHINode *> seenPhi1;
            if (lValuesAreEqual(elements[NonNull], elements[i], seenPhi0, seenPhi1) == false) {
                return NULL;
            }
        }
        return elements[NonNull];
    }

    // Catch a pattern of broadcast implemented as InsertElement + Shuffle:
    //   %broadcast_init.0 = insertelement <4 x i32> undef, i32 %val, i32 0
    //   %broadcast.1 = shufflevector <4 x i32> %smear.0, <4 x i32> undef,
    //                                              <4 x i32> zeroinitializer
    // Or:
    //   %gep_ptr2int_broadcast_init = insertelement <8 x i64> undef, i64 %gep_ptr2int, i32 0
    //   %0 = add <8 x i64> %gep_ptr2int_broadcast_init,
    //              <i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>
    //   %gep_offset = shufflevector <8 x i64> %0, <8 x i64> undef, <8 x i32> zeroinitializer
    else if (llvm::ShuffleVectorInst *shuf = llvm::dyn_cast<llvm::ShuffleVectorInst>(inst)) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
        llvm::Value *indices = shuf->getShuffleMaskForBitcode();
#else
        llvm::Value *indices = shuf->getOperand(2);
#endif

        if (llvm::isa<llvm::ConstantAggregateZero>(indices)) {
            llvm::Value *op = shuf->getOperand(0);
            llvm::InsertElementInst *ie = llvm::dyn_cast<llvm::InsertElementInst>(op);
            if (ie == NULL && searchFirstUndef) {
                // Trying to recognize 2nd pattern
                llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(op);
                if (bop != NULL && ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop))) {
                    if (lIsFirstElementConstVector(bop->getOperand(1))) {
                        ie = llvm::dyn_cast<llvm::InsertElementInst>(bop->getOperand(0));
                    } else if (llvm::isa<llvm::InsertElementInst>(bop->getOperand(1))) {
                        // Or shuffle vector can accept insertelement itself
                        ie = llvm::cast<llvm::InsertElementInst>(bop->getOperand(1));
                    }
                }
            }
            if (ie != NULL && llvm::isa<llvm::UndefValue>(ie->getOperand(0))) {
                llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(ie->getOperand(2));
                if (ci->isZero()) {
                    return ie->getOperand(1);
                }
            }
        }
    }
    return NULL;
}

bool LLVMExtractVectorInts(llvm::Value *v, int64_t ret[], int *nElts) {
    // Make sure we do in fact have a vector of integer values here
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
#else
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(v->getType());
#endif
    Assert(vt != NULL);
    Assert(llvm::isa<llvm::IntegerType>(vt->getElementType()));

    *nElts = (int)vt->getNumElements();

    if (llvm::isa<llvm::ConstantAggregateZero>(v)) {
        for (int i = 0; i < (int)vt->getNumElements(); ++i)
            ret[i] = 0;
        return true;
    }

    llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(v);
    if (cv == NULL)
        return false;

    for (int i = 0; i < (int)cv->getNumElements(); ++i)
        ret[i] = cv->getElementAsInteger(i);
    return true;
}

static bool lVectorValuesAllEqual(llvm::Value *v, int vectorLength, std::vector<llvm::PHINode *> &seenPhis,
                                  llvm::Value **splatValue = NULL);

/** This function checks to see if the given (scalar or vector) value is an
    exact multiple of baseValue.  It returns true if so, and false if not
    (or if it's not able to determine if it is).  Any vector value passed
    in is required to have the same value in all elements (so that we can
    just check the first element to be a multiple of the given value.)
 */
static bool lIsExactMultiple(llvm::Value *val, int baseValue, int vectorLength,
                             std::vector<llvm::PHINode *> &seenPhis) {
    if (llvm::isa<llvm::VectorType>(val->getType()) == false) {
        // If we've worked down to a constant int, then the moment of truth
        // has arrived...
        llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(val);
        if (ci != NULL)
            return (ci->getZExtValue() % baseValue) == 0;
    } else
        Assert(LLVMVectorValuesAllEqual(val));

    if (llvm::isa<llvm::InsertElementInst>(val) || llvm::isa<llvm::ShuffleVectorInst>(val)) {
        llvm::Value *element = LLVMFlattenInsertChain(val, g->target->getVectorWidth());
        // We just need to check the scalar first value, since we know that
        // all elements are equal
        return element ? lIsExactMultiple(element, baseValue, vectorLength, seenPhis) : false;
    }

    llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(val);
    if (phi != NULL) {
        for (unsigned int i = 0; i < seenPhis.size(); ++i)
            if (phi == seenPhis[i])
                return true;

        seenPhis.push_back(phi);
        unsigned int numIncoming = phi->getNumIncomingValues();

        // Check all of the incoming values: if all of them pass, then
        // we're good.
        for (unsigned int i = 0; i < numIncoming; ++i) {
            llvm::Value *incoming = phi->getIncomingValue(i);
            bool mult = lIsExactMultiple(incoming, baseValue, vectorLength, seenPhis);
            if (mult == false) {
                seenPhis.pop_back();
                return false;
            }
        }
        seenPhis.pop_back();
        return true;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(val);
    if (bop != NULL && ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop))) {
        llvm::Value *op0 = bop->getOperand(0);
        llvm::Value *op1 = bop->getOperand(1);

        bool be0 = lIsExactMultiple(op0, baseValue, vectorLength, seenPhis);
        bool be1 = lIsExactMultiple(op1, baseValue, vectorLength, seenPhis);
        return (be0 && be1);
    }
    // FIXME: mul? casts? ... ?

    return false;
}

/** Returns the next power of two greater than or equal to the given
    value. */
static int lRoundUpPow2(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/** Try to determine if all of the elements of the given vector value have
    the same value when divided by the given baseValue.  The function
    returns true if this can be determined to be the case, and false
    otherwise.  (This function may fail to identify some cases where it
    does in fact have this property, but should never report a given value
    as being a multiple if it isn't!)
 */
static bool lAllDivBaseEqual(llvm::Value *val, int64_t baseValue, int vectorLength,
                             std::vector<llvm::PHINode *> &seenPhis, bool &canAdd) {
    Assert(llvm::isa<llvm::VectorType>(val->getType()));
    // Make sure the base value is a positive power of 2
    Assert(baseValue > 0 && (baseValue & (baseValue - 1)) == 0);

    // The easy case
    if (lVectorValuesAllEqual(val, vectorLength, seenPhis))
        return true;

    int64_t vecVals[ISPC_MAX_NVEC];
    int nElts;
    if (llvm::isa<llvm::VectorType>(val->getType()) && LLVMExtractVectorInts(val, vecVals, &nElts)) {
        // If we have a vector of compile-time constant integer values,
        // then go ahead and check them directly..
        int64_t firstDiv = vecVals[0] / baseValue;
        for (int i = 1; i < nElts; ++i)
            if ((vecVals[i] / baseValue) != firstDiv)
                return false;

        return true;
    }

    llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(val);
    if (phi != NULL) {
        for (unsigned int i = 0; i < seenPhis.size(); ++i)
            if (phi == seenPhis[i])
                return true;

        seenPhis.push_back(phi);
        unsigned int numIncoming = phi->getNumIncomingValues();

        // Check all of the incoming values: if all of them pass, then
        // we're good.
        for (unsigned int i = 0; i < numIncoming; ++i) {
            llvm::Value *incoming = phi->getIncomingValue(i);
            bool ca = canAdd;
            bool mult = lAllDivBaseEqual(incoming, baseValue, vectorLength, seenPhis, ca);
            if (mult == false) {
                seenPhis.pop_back();
                return false;
            }
        }
        seenPhis.pop_back();
        return true;
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(val);
    if (bop != NULL && ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) && canAdd == true) {
        llvm::Value *op0 = bop->getOperand(0);
        llvm::Value *op1 = bop->getOperand(1);

        // Otherwise we're only going to worry about the following case,
        // which comes up often when looping over SOA data:
        // ashr %val, <constant shift>
        // where %val = add %smear, <0,1,2,3...>
        // and where the maximum of the <0,...> vector in the add is less than
        //   1<<(constant shift),
        // and where %smear is a smear of a value that is a multiple of
        //   baseValue.

        int64_t addConstants[ISPC_MAX_NVEC];
        if (LLVMExtractVectorInts(op1, addConstants, &nElts) == false)
            return false;
        Assert(nElts == vectorLength);

        // Do all of them give the same value when divided by baseValue?
        int64_t firstConstDiv = addConstants[0] / baseValue;
        for (int i = 1; i < vectorLength; ++i)
            if ((addConstants[i] / baseValue) != firstConstDiv)
                return false;

        if (lVectorValuesAllEqual(op0, vectorLength, seenPhis) == false)
            return false;

        // Note that canAdd is a reference parameter; setting this ensures
        // that we don't allow multiple adds in other parts of the chain of
        // dependent values from here.
        canAdd = false;

        // Now we need to figure out the required alignment (in numbers of
        // elements of the underlying type being indexed) of the value to
        // which these integer addConstant[] values are being added to.  We
        // know that we have addConstant[] values that all give the same
        // value when divided by baseValue, but we may need a less-strict
        // alignment than baseValue depending on the actual values.
        //
        // As an example, consider a case where the baseValue alignment is
        // 16, but the addConstants here are <0,1,2,3>.  In that case, the
        // value to which addConstants is added to only needs to be a
        // multiple of 4.  Conversely, if addConstants are <4,5,6,7>, then
        // we need a multiple of 8 to ensure that the final added result
        // will still have the same value for all vector elements when
        // divided by baseValue.
        //
        // All that said, here we need to find the maximum value of any of
        // the addConstants[], mod baseValue.  If we round that up to the
        // next power of 2, we'll have a value that will be no greater than
        // baseValue and sometimes less.
        int maxMod = int(addConstants[0] % baseValue);
        for (int i = 1; i < vectorLength; ++i)
            maxMod = std::max(maxMod, int(addConstants[i] % baseValue));
        int requiredAlignment = lRoundUpPow2(maxMod);

        std::vector<llvm::PHINode *> seenPhisEEM;
        return lIsExactMultiple(op0, requiredAlignment, vectorLength, seenPhisEEM);
    }
    // TODO: could handle mul by a vector of equal constant integer values
    // and the like here and adjust the 'baseValue' value when it evenly
    // divides, but unclear if it's worthwhile...

    return false;
}

/** Given a vector shift right of some value by some amount, try to
    determine if all of the elements of the final result have the same
    value (i.e. whether the high bits are all equal, disregarding the low
    bits that are shifted out.)  Returns true if so, and false otherwise.
 */
static bool lVectorShiftRightAllEqual(llvm::Value *val, llvm::Value *shift, int vectorLength) {
    // Are we shifting all elements by a compile-time constant amount?  If
    // not, give up.
    int64_t shiftAmount[ISPC_MAX_NVEC];
    int nElts;
    if (LLVMExtractVectorInts(shift, shiftAmount, &nElts) == false)
        return false;
    Assert(nElts == vectorLength);

    // Is it the same amount for all elements?
    for (int i = 0; i < vectorLength; ++i)
        if (shiftAmount[i] != shiftAmount[0])
            return false;

    // Now see if the value divided by (1 << shift) can be determined to
    // have the same value for all vector elements.
    int pow2 = 1 << shiftAmount[0];
    bool canAdd = true;
    std::vector<llvm::PHINode *> seenPhis;
    bool eq = lAllDivBaseEqual(val, pow2, vectorLength, seenPhis, canAdd);
#if 0
    fprintf(stderr, "check all div base equal:\n");
    LLVMDumpValue(shift);
    LLVMDumpValue(val);
    fprintf(stderr, "----> %s\n\n", eq ? "true" : "false");
#endif
    return eq;
}

static bool lVectorValuesAllEqual(llvm::Value *v, int vectorLength, std::vector<llvm::PHINode *> &seenPhis,
                                  llvm::Value **splatValue) {
    if (vectorLength == 1)
        return true;

    if (llvm::isa<llvm::ConstantAggregateZero>(v)) {
        if (splatValue) {
            llvm::ConstantAggregateZero *caz = llvm::dyn_cast<llvm::ConstantAggregateZero>(v);
            *splatValue = caz->getSequentialElement();
        }
        return true;
    }

    llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v);
    if (cv != NULL) {
        llvm::Value *splat = cv->getSplatValue();
        if (splat != NULL && splatValue) {
            *splatValue = splat;
        }
        return (splat != NULL);
    }

    llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v);
    if (cdv != NULL) {
        llvm::Value *splat = cdv->getSplatValue();
        if (splat != NULL && splatValue) {
            *splatValue = splat;
        }
        return (splat != NULL);
    }

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v);
    if (bop != NULL) {
        // Easy case: both operands are all equal -> return true
        if (lVectorValuesAllEqual(bop->getOperand(0), vectorLength, seenPhis) &&
            lVectorValuesAllEqual(bop->getOperand(1), vectorLength, seenPhis))
            return true;

        // If it's a shift, take a special path that tries to check if the
        // high (surviving) bits of the values are equal.
        if (bop->getOpcode() == llvm::Instruction::AShr || bop->getOpcode() == llvm::Instruction::LShr)
            return lVectorShiftRightAllEqual(bop->getOperand(0), bop->getOperand(1), vectorLength);

        return false;
    }

    llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(v);
    if (cast != NULL)
        return lVectorValuesAllEqual(cast->getOperand(0), vectorLength, seenPhis);

    llvm::InsertElementInst *ie = llvm::dyn_cast<llvm::InsertElementInst>(v);
    if (ie != NULL) {
        return (LLVMFlattenInsertChain(ie, vectorLength) != NULL);
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
            if (!lVectorValuesAllEqual(phi->getIncomingValue(i), vectorLength, seenPhis)) {
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

    if (llvm::isa<llvm::CallInst>(v) || llvm::isa<llvm::LoadInst>(v) || !llvm::isa<llvm::Instruction>(v))
        return false;

    llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(v);
    if (shuffle != NULL) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
        llvm::Value *indices = shuffle->getShuffleMaskForBitcode();
#else
        llvm::Value *indices = shuffle->getOperand(2);
#endif

        if (lVectorValuesAllEqual(indices, vectorLength, seenPhis))
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

/** Tests to see if all of the elements of the vector in the 'v' parameter
    are equal.  This is a conservative test and may return false for arrays
    where the values are actually all equal.
*/
bool LLVMVectorValuesAllEqual(llvm::Value *v, llvm::Value **splat) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
#else
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(v->getType());
#endif
    Assert(vt != NULL);
    int vectorLength = vt->getNumElements();

    std::vector<llvm::PHINode *> seenPhis;
    bool equal = lVectorValuesAllEqual(v, vectorLength, seenPhis, splat);

    Debug(SourcePos(), "LLVMVectorValuesAllEqual(%s) -> %s.", v->getName().str().c_str(), equal ? "true" : "false");
#ifndef ISPC_NO_DUMPS
    if (g->debugPrint)
        LLVMDumpValue(v);
#endif

    return equal;
}

/** Tests to see if a binary operator has an OR which is equivalent to an ADD.*/
bool IsOrEquivalentToAdd(llvm::Value *op) {
    bool isEq = false;
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(op);
    if (bop != NULL && bop->getOpcode() == llvm::Instruction::Or) {
        // Special case when A+B --> A|B transformation is triggered
        // We need to prove that A|B == A+B
        llvm::Module *module = bop->getParent()->getParent()->getParent();
        llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
        if (!haveNoCommonBitsSet(op0, op1, module->getDataLayout()) == false) {
            // Fallback to A+B case
            isEq = true;
        }
    }
    return isEq;
}

static bool lVectorIsLinear(llvm::Value *v, int vectorLength, int stride, std::vector<llvm::PHINode *> &seenPhis);

/** Given a vector of compile-time constant integer values, test to see if
    they are a linear sequence of constant integers starting from an
    arbirary value but then having a step of value "stride" between
    elements.
 */
static bool lVectorIsLinearConstantInts(llvm::ConstantDataVector *cv, int vectorLength, int stride) {
    // Flatten the vector out into the elements array
    llvm::SmallVector<llvm::Constant *, ISPC_MAX_NVEC> elements;
    for (int i = 0; i < (int)cv->getNumElements(); ++i)
        elements.push_back(cv->getElementAsConstant(i));
    Assert((int)elements.size() == vectorLength);

    llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(elements[0]);
    if (ci == NULL)
        // Not a vector of integers
        return false;

    int64_t prevVal = ci->getSExtValue();

    // For each element in the array, see if it is both a ConstantInt and
    // if the difference between it and the value of the previous element
    // is stride.  If not, fail.
    for (int i = 1; i < vectorLength; ++i) {
        ci = llvm::dyn_cast<llvm::ConstantInt>(elements[i]);
        if (ci == NULL)
            return false;

        int64_t nextVal = ci->getSExtValue();
        if (prevVal + stride != nextVal)
            return false;

        prevVal = nextVal;
    }
    return true;
}

/** Checks to see if (op0 * op1) is a linear vector where the result is a
    vector with values that increase by stride.
 */
static bool lCheckMulForLinear(llvm::Value *op0, llvm::Value *op1, int vectorLength, int stride,
                               std::vector<llvm::PHINode *> &seenPhis) {
    // Is the first operand a constant integer value splatted across all of
    // the lanes?
    llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(op0);
    if (cv == NULL)
        return false;

    llvm::Constant *csplat = cv->getSplatValue();
    if (csplat == NULL)
        return false;

    llvm::ConstantInt *splat = llvm::dyn_cast<llvm::ConstantInt>(csplat);
    if (splat == NULL)
        return false;

    // If the splat value doesn't evenly divide the stride we're looking
    // for, there's no way that we can get the linear sequence we're
    // looking or.
    int64_t splatVal = splat->getSExtValue();
    if (splatVal == 0 || splatVal > stride || (stride % splatVal) != 0)
        return false;

    // Check to see if the other operand is a linear vector with stride
    // given by stride/splatVal.
    return lVectorIsLinear(op1, vectorLength, (int)(stride / splatVal), seenPhis);
}

/** Checks to see if (op0 << op1) is a linear vector where the result is a
    vector with values that increase by stride.
 */
static bool lCheckShlForLinear(llvm::Value *op0, llvm::Value *op1, int vectorLength, int stride,
                               std::vector<llvm::PHINode *> &seenPhis) {
    // Is the second operand a constant integer value splatted across all of
    // the lanes?
    llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(op1);
    if (cv == NULL)
        return false;

    llvm::Constant *csplat = cv->getSplatValue();
    if (csplat == NULL)
        return false;

    llvm::ConstantInt *splat = llvm::dyn_cast<llvm::ConstantInt>(csplat);
    if (splat == NULL)
        return false;

    // If (1 << the splat value) doesn't evenly divide the stride we're
    // looking for, there's no way that we can get the linear sequence
    // we're looking or.
    int64_t equivalentMul = (1LL << splat->getSExtValue());
    if (equivalentMul > stride || (stride % equivalentMul) != 0)
        return false;

    // Check to see if the first operand is a linear vector with stride
    // given by stride/splatVal.
    return lVectorIsLinear(op0, vectorLength, (int)(stride / equivalentMul), seenPhis);
}

/** Given (op0 AND op1), try and see if we can determine if the result is a
    linear sequence with a step of "stride" between values.  Returns true
    if so and false otherwise.  This pattern comes up when accessing SOA
    data.
 */
static bool lCheckAndForLinear(llvm::Value *op0, llvm::Value *op1, int vectorLength, int stride,
                               std::vector<llvm::PHINode *> &seenPhis) {
    // Require op1 to be a compile-time constant
    int64_t maskValue[ISPC_MAX_NVEC];
    int nElts;
    if (LLVMExtractVectorInts(op1, maskValue, &nElts) == false)
        return false;
    Assert(nElts == vectorLength);

    // Is op1 a smear of the same value across all lanes?  Give up if not.
    for (int i = 1; i < vectorLength; ++i)
        if (maskValue[i] != maskValue[0])
            return false;

    // If the op1 value isn't power of 2 minus one, then also give up.
    int64_t maskPlusOne = maskValue[0] + 1;
    bool isPowTwo = (maskPlusOne & (maskPlusOne - 1)) == 0;
    if (isPowTwo == false)
        return false;

    // The case we'll covert here is op0 being a linear vector with desired
    // stride, and where all of the values of op0, when divided by
    // maskPlusOne, have the same value.
    if (lVectorIsLinear(op0, vectorLength, stride, seenPhis) == false)
        return false;

    bool canAdd = true;
    bool isMult = lAllDivBaseEqual(op0, maskPlusOne, vectorLength, seenPhis, canAdd);
    return isMult;
}

static bool lVectorIsLinear(llvm::Value *v, int vectorLength, int stride, std::vector<llvm::PHINode *> &seenPhis) {
    // First try the easy case: if the values are all just constant
    // integers and have the expected stride between them, then we're done.
    llvm::ConstantDataVector *cv = llvm::dyn_cast<llvm::ConstantDataVector>(v);
    if (cv != NULL)
        return lVectorIsLinearConstantInts(cv, vectorLength, stride);

    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v);
    if (bop != NULL) {
        // FIXME: is it right to pass the seenPhis to the all equal check as well??
        llvm::Value *op0 = bop->getOperand(0), *op1 = bop->getOperand(1);
        if ((bop->getOpcode() == llvm::Instruction::Add) || IsOrEquivalentToAdd(bop)) {
            // There are two cases to check if we have an add:
            //
            // programIndex + unif -> ascending linear seqeuence
            // unif + programIndex -> ascending linear sequence
            bool l0 = lVectorIsLinear(op0, vectorLength, stride, seenPhis);
            bool e1 = lVectorValuesAllEqual(op1, vectorLength, seenPhis);
            if (l0 && e1)
                return true;

            bool e0 = lVectorValuesAllEqual(op0, vectorLength, seenPhis);
            bool l1 = lVectorIsLinear(op1, vectorLength, stride, seenPhis);
            return (e0 && l1);
        } else if (bop->getOpcode() == llvm::Instruction::Sub)
            // For subtraction, we only match:
            // programIndex - unif -> ascending linear seqeuence
            return (lVectorIsLinear(bop->getOperand(0), vectorLength, stride, seenPhis) &&
                    lVectorValuesAllEqual(bop->getOperand(1), vectorLength, seenPhis));
        else if (bop->getOpcode() == llvm::Instruction::Mul) {
            // Multiplies are a bit trickier, so are handled in a separate
            // function.
            bool m0 = lCheckMulForLinear(op0, op1, vectorLength, stride, seenPhis);
            if (m0)
                return true;
            bool m1 = lCheckMulForLinear(op1, op0, vectorLength, stride, seenPhis);
            return m1;
        } else if (bop->getOpcode() == llvm::Instruction::Shl) {
            // Sometimes multiplies come in as shift lefts (especially in
            // LLVM 3.4+).
            bool linear = lCheckShlForLinear(op0, op1, vectorLength, stride, seenPhis);
            return linear;
        } else if (bop->getOpcode() == llvm::Instruction::And) {
            // Special case for some AND-related patterns that come up when
            // looping over SOA data
            bool linear = lCheckAndForLinear(op0, op1, vectorLength, stride, seenPhis);
            return linear;
        } else
            return false;
    }

    llvm::CastInst *ci = llvm::dyn_cast<llvm::CastInst>(v);
    if (ci != NULL)
        return lVectorIsLinear(ci->getOperand(0), vectorLength, stride, seenPhis);

    if (llvm::isa<llvm::CallInst>(v) || llvm::isa<llvm::LoadInst>(v))
        return false;

    llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(v);
    if (phi != NULL) {
        for (unsigned int i = 0; i < seenPhis.size(); ++i)
            if (seenPhis[i] == phi)
                return true;

        seenPhis.push_back(phi);

        unsigned int numIncoming = phi->getNumIncomingValues();
        // Check all of the incoming values: if all of them are all equal,
        // then we're good.
        for (unsigned int i = 0; i < numIncoming; ++i) {
            if (!lVectorIsLinear(phi->getIncomingValue(i), vectorLength, stride, seenPhis)) {
                seenPhis.pop_back();
                return false;
            }
        }

        seenPhis.pop_back();
        return true;
    }

    // TODO: is any reason to worry about these?
    if (llvm::isa<llvm::InsertElementInst>(v))
        return false;

    // TODO: we could also handle shuffles, but we haven't yet seen any
    // cases where doing so would detect cases where actually have a linear
    // vector.
    llvm::ShuffleVectorInst *shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(v);
    if (shuffle != NULL)
        return false;

#if 0
    fprintf(stderr, "linear check: ");
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

/** Given vector of integer-typed values, see if the elements of the array
    have a step of 'stride' between their values.  This function tries to
    handle as many possibilities as possible, including things like all
    elements equal to some non-constant value plus an integer offset, etc.
*/
bool LLVMVectorIsLinear(llvm::Value *v, int stride) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
#else
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(v->getType());
#endif
    Assert(vt != NULL);
    int vectorLength = vt->getNumElements();

    std::vector<llvm::PHINode *> seenPhis;
    bool linear = lVectorIsLinear(v, vectorLength, stride, seenPhis);
    Debug(SourcePos(), "LLVMVectorIsLinear(%s) -> %s.", v->getName().str().c_str(), linear ? "true" : "false");
#ifndef ISPC_NO_DUMPS
    if (g->debugPrint)
        LLVMDumpValue(v);
#endif

    return linear;
}

#ifndef ISPC_NO_DUMPS
static void lDumpValue(llvm::Value *v, std::set<llvm::Value *> &done) {
    if (done.find(v) != done.end())
        return;

    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    if (done.size() > 0 && inst == NULL)
        return;

    fprintf(stderr, "  ");
    v->dump();
    done.insert(v);

    if (inst == NULL)
        return;

    for (unsigned i = 0; i < inst->getNumOperands(); ++i)
        lDumpValue(inst->getOperand(i), done);
}

void LLVMDumpValue(llvm::Value *v) {
    std::set<llvm::Value *> done;
    lDumpValue(v, done);
    fprintf(stderr, "----\n");
}
#endif

static llvm::Value *lExtractFirstVectorElement(llvm::Value *v, std::map<llvm::PHINode *, llvm::PHINode *> &phiMap) {
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(v->getType());
#else
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(v->getType());
#endif
    Assert(vt != NULL);

    // First, handle various constant types; do the extraction manually, as
    // appropriate.
    if (llvm::isa<llvm::ConstantAggregateZero>(v) == true) {
        return llvm::Constant::getNullValue(vt->getElementType());
    }
    if (llvm::ConstantVector *cv = llvm::dyn_cast<llvm::ConstantVector>(v)) {
        return cv->getOperand(0);
    }
    if (llvm::ConstantDataVector *cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v))
        return cdv->getElementAsConstant(0);

    // Otherwise, all that we should have at this point is an instruction
    // of some sort
    Assert(llvm::isa<llvm::Constant>(v) == false);
    Assert(llvm::isa<llvm::Instruction>(v) == true);

    std::string newName = v->getName().str() + std::string(".elt0");

    // Rewrite regular binary operators and casts to the scalarized
    // equivalent.
    llvm::BinaryOperator *bop = llvm::dyn_cast<llvm::BinaryOperator>(v);
    if (bop != NULL) {
        llvm::Value *v0 = lExtractFirstVectorElement(bop->getOperand(0), phiMap);
        llvm::Value *v1 = lExtractFirstVectorElement(bop->getOperand(1), phiMap);
        Assert(v0 != NULL);
        Assert(v1 != NULL);
        // Note that the new binary operator is inserted immediately before
        // the previous vector one
        return llvm::BinaryOperator::Create(bop->getOpcode(), v0, v1, newName, bop);
    }

    llvm::CastInst *cast = llvm::dyn_cast<llvm::CastInst>(v);
    if (cast != NULL) {
        llvm::Value *v = lExtractFirstVectorElement(cast->getOperand(0), phiMap);
        // Similarly, the equivalent scalar cast instruction goes right
        // before the vector cast
        return llvm::CastInst::Create(cast->getOpcode(), v, vt->getElementType(), newName, cast);
    }

    llvm::PHINode *phi = llvm::dyn_cast<llvm::PHINode>(v);
    if (phi != NULL) {
        // For PHI notes, recursively scalarize them.
        if (phiMap.find(phi) != phiMap.end())
            return phiMap[phi];

        // We need to create the new scalar PHI node immediately, though,
        // and put it in the map<>, so that if we come back to this node
        // via a recursive lExtractFirstVectorElement() call, then we can
        // return the pointer and not get stuck in an infinite loop.
        //
        // The insertion point for the new phi node also has to be the
        // start of the bblock of the original phi node.

        llvm::Instruction *phiInsertPos = &*(phi->getParent()->begin());
        llvm::PHINode *scalarPhi =
            llvm::PHINode::Create(vt->getElementType(), phi->getNumIncomingValues(), newName, phiInsertPos);
        phiMap[phi] = scalarPhi;

        for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
            llvm::Value *v = lExtractFirstVectorElement(phi->getIncomingValue(i), phiMap);
            scalarPhi->addIncoming(v, phi->getIncomingBlock(i));
        }

        return scalarPhi;
    }

    // We should consider "shuffle" case and "insertElement" case separately.
    // For example we can have shuffle(mul, undef, zero) but function
    // "LLVMFlattenInsertChain" can handle only case shuffle(insertElement, undef, zero).
    // Also if we have insertElement under shuffle we will handle it the next call of
    // "lExtractFirstVectorElement" function.
    if (llvm::isa<llvm::ShuffleVectorInst>(v)) {
        llvm::ShuffleVectorInst *shuf = llvm::dyn_cast<llvm::ShuffleVectorInst>(v);
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
        llvm::Value *indices = shuf->getShuffleMaskForBitcode();
#else
        llvm::Value *indices = shuf->getOperand(2);
#endif
        if (llvm::isa<llvm::ConstantAggregateZero>(indices)) {
            return lExtractFirstVectorElement(shuf->getOperand(0), phiMap);
        }
    }

    // If we have a chain of insertelement instructions, then we can just
    // flatten them out and grab the value for the first one.
    if (llvm::isa<llvm::InsertElementInst>(v)) {
        return LLVMFlattenInsertChain(v, vt->getNumElements(), false);
    }

    // Worst case, for everything else, just do a regular extract element
    // instruction, which we insert immediately after the instruction we
    // have here.
    llvm::Instruction *insertAfter = llvm::dyn_cast<llvm::Instruction>(v);
    Assert(insertAfter != NULL);
    llvm::Instruction *ee = llvm::ExtractElementInst::Create(v, LLVMInt32(0), "first_elt", (llvm::Instruction *)NULL);
    ee->insertAfter(insertAfter);
    return ee;
}

llvm::Value *LLVMExtractFirstVectorElement(llvm::Value *v) {
    std::map<llvm::PHINode *, llvm::PHINode *> phiMap;
    llvm::Value *ret = lExtractFirstVectorElement(v, phiMap);
    return ret;
}

/** Given two vectors of the same type, concatenate them into a vector that
    has twice as many elements, where the first half has the elements from
    the first vector and the second half has the elements from the second
    vector.
 */
llvm::Value *LLVMConcatVectors(llvm::Value *v1, llvm::Value *v2, llvm::Instruction *insertBefore) {
    Assert(v1->getType() == v2->getType());
#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
    llvm::FixedVectorType *vt = llvm::dyn_cast<llvm::FixedVectorType>(v1->getType());
#else
    llvm::VectorType *vt = llvm::dyn_cast<llvm::VectorType>(v1->getType());
#endif
    Assert(vt != NULL);

    int32_t identity[ISPC_MAX_NVEC];
    int resultSize = 2 * vt->getNumElements();
    Assert(resultSize <= ISPC_MAX_NVEC);
    for (int i = 0; i < resultSize; ++i)
        identity[i] = i;

    return LLVMShuffleVectors(v1, v2, identity, resultSize, insertBefore);
}

/** Shuffle two vectors together with a ShuffleVectorInst, returning a
    vector with shufSize elements, where the shuf[] array offsets are used
    to determine which element from the two given vectors is used for each
    result element. */
llvm::Value *LLVMShuffleVectors(llvm::Value *v1, llvm::Value *v2, int32_t shuf[], int shufSize,
                                llvm::Instruction *insertBefore) {
    std::vector<llvm::Constant *> shufVec;
    for (int i = 0; i < shufSize; ++i) {
        if (shuf[i] == -1)
            shufVec.push_back(llvm::UndefValue::get(LLVMTypes::Int32Type));
        else
            shufVec.push_back(LLVMInt32(shuf[i]));
    }

    llvm::ArrayRef<llvm::Constant *> aref(&shufVec[0], &shufVec[shufSize]);
    llvm::Value *vec = llvm::ConstantVector::get(aref);

    return new llvm::ShuffleVectorInst(v1, v2, vec, "shuffle", insertBefore);
}

#ifdef ISPC_GENX_ENABLED
bool lIsSVMLoad(llvm::Instruction *inst) {
    Assert(inst);

    switch (llvm::GenXIntrinsic::getGenXIntrinsicID(inst)) {
    case llvm::GenXIntrinsic::genx_svm_block_ld:
    case llvm::GenXIntrinsic::genx_svm_block_ld_unaligned:
    case llvm::GenXIntrinsic::genx_svm_gather:
        return true;
    default:
        return false;
    }
}

void lGetAddressSpace(llvm::Value *v, std::set<llvm::Value *> &done, std::set<AddressSpace> &addrSpaceVec) {
    if (done.find(v) != done.end()) {
        if (llvm::isa<llvm::PointerType>(v->getType()))
            addrSpaceVec.insert(AddressSpace::External);
        return;
    }
    // Found global value
    if (llvm::isa<llvm::GlobalValue>(v)) {
        addrSpaceVec.insert(AddressSpace::Global);
        return;
    }

    llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(v);
    bool isConstExpr = false;
    if (inst == NULL) {
        // Case when GEP is constant expression
        if (llvm::ConstantExpr *constExpr = llvm::dyn_cast<llvm::ConstantExpr>(v)) {
            // This instruction isn't inserted anywhere, so delete it when done
            inst = constExpr->getAsInstruction();
            isConstExpr = true;
        }
    }

    if (done.size() > 0 && inst == NULL) {
        // Found external pointer like "float* %aFOO"
        if (llvm::isa<llvm::PointerType>(v->getType()))
            addrSpaceVec.insert(AddressSpace::External);
        return;
    }

    done.insert(v);

    // Found value allocated on stack like "%val = alloca [16 x float]"
    if (llvm::isa<llvm::AllocaInst>(v)) {
        addrSpaceVec.insert(AddressSpace::Local);
        return;
    }

    if (inst == NULL || llvm::isa<llvm::CallInst>(v)) {
        if (llvm::isa<llvm::PointerType>(v->getType()) || (inst && lIsSVMLoad(inst)))
            addrSpaceVec.insert(AddressSpace::External);
        return;
    }
    // For GEP we check only pointer operand, for the rest check all
    if (llvm::isa<llvm::GetElementPtrInst>(inst)) {
        lGetAddressSpace(inst->getOperand(0), done, addrSpaceVec);
    } else {
        for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
            lGetAddressSpace(inst->getOperand(i), done, addrSpaceVec);
        }
    }

    if (isConstExpr) {
        // This is the only return point that constant expression instruction
        // can reach, drop all references here
        inst->dropAllReferences();
    }
}

/** This routine attempts to determine if the given value is pointing to
    stack-allocated memory. The basic strategy is to traverse through the
    operands and see if the pointer originally comes from an AllocaInst.
*/
AddressSpace GetAddressSpace(llvm::Value *v) {
    std::set<llvm::Value *> done;
    std::set<AddressSpace> addrSpaceVec;
    lGetAddressSpace(v, done, addrSpaceVec);
    if (addrSpaceVec.find(AddressSpace::External) != addrSpaceVec.end()) {
        return AddressSpace::External;
    }
    if (addrSpaceVec.find(AddressSpace::Global) != addrSpaceVec.end()) {
        return AddressSpace::Global;
    }
    return AddressSpace::Local;
}
#endif
} // namespace ispc
