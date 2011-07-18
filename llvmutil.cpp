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
#include "type.h"

LLVM_TYPE_CONST llvm::Type *LLVMTypes::VoidType = NULL;
LLVM_TYPE_CONST llvm::PointerType *LLVMTypes::VoidPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::BoolType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int8Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int16Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int32Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int32PointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int64Type = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int64PointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::FloatType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::FloatPointerType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::DoubleType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::DoublePointerType = NULL;

LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::MaskType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::BoolVectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int1VectorType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int32VectorType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int32VectorPointerType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::Int64VectorType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::Int64VectorPointerType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::FloatVectorType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::FloatVectorPointerType = NULL;
LLVM_TYPE_CONST llvm::VectorType *LLVMTypes::DoubleVectorType = NULL;
LLVM_TYPE_CONST llvm::Type *LLVMTypes::DoubleVectorPointerType = NULL;
LLVM_TYPE_CONST llvm::ArrayType *LLVMTypes::VoidPointerVectorType = NULL;

llvm::Constant *LLVMTrue = NULL;
llvm::Constant *LLVMFalse = NULL;
llvm::Constant *LLVMMaskAllOn = NULL;
llvm::Constant *LLVMMaskAllOff = NULL;


void
InitLLVMUtil(llvm::LLVMContext *ctx, Target target) {
    LLVMTypes::VoidType = llvm::Type::getVoidTy(*ctx);
    LLVMTypes::VoidPointerType = llvm::PointerType::get(llvm::Type::getInt8Ty(*ctx), 0);
    LLVMTypes::BoolType = llvm::Type::getInt1Ty(*ctx);
    LLVMTypes::Int8Type = llvm::Type::getInt8Ty(*ctx);
    LLVMTypes::Int16Type = llvm::Type::getInt16Ty(*ctx);
    LLVMTypes::Int32Type = llvm::Type::getInt32Ty(*ctx);
    LLVMTypes::Int32PointerType = llvm::PointerType::get(LLVMTypes::Int32Type, 0);
    LLVMTypes::Int64Type = llvm::Type::getInt64Ty(*ctx);
    LLVMTypes::Int64PointerType = llvm::PointerType::get(LLVMTypes::Int64Type, 0);
    LLVMTypes::FloatType = llvm::Type::getFloatTy(*ctx);
    LLVMTypes::FloatPointerType = llvm::PointerType::get(LLVMTypes::FloatType, 0);
    LLVMTypes::DoubleType = llvm::Type::getDoubleTy(*ctx);
    LLVMTypes::DoublePointerType = llvm::PointerType::get(LLVMTypes::DoubleType, 0);

    // Note that both the mask and bool vectors are vector of int32s
    // (not i1s).  LLVM ends up generating much better SSE code with
    // this representation.
    LLVMTypes::MaskType = LLVMTypes::BoolVectorType =
        llvm::VectorType::get(llvm::Type::getInt32Ty(*ctx), target.vectorWidth);

    LLVMTypes::Int1VectorType = 
        llvm::VectorType::get(llvm::Type::getInt1Ty(*ctx), target.vectorWidth);
    LLVMTypes::Int32VectorType = 
        llvm::VectorType::get(LLVMTypes::Int32Type, target.vectorWidth);
    LLVMTypes::Int32VectorPointerType = llvm::PointerType::get(LLVMTypes::Int32VectorType, 0);
    LLVMTypes::Int64VectorType = 
        llvm::VectorType::get(LLVMTypes::Int64Type, target.vectorWidth);
    LLVMTypes::Int64VectorPointerType = llvm::PointerType::get(LLVMTypes::Int64VectorType, 0);
    LLVMTypes::FloatVectorType = 
        llvm::VectorType::get(LLVMTypes::FloatType, target.vectorWidth);
    LLVMTypes::FloatVectorPointerType = llvm::PointerType::get(LLVMTypes::FloatVectorType, 0);
    LLVMTypes::DoubleVectorType = 
        llvm::VectorType::get(LLVMTypes::DoubleType, target.vectorWidth);
    LLVMTypes::DoubleVectorPointerType = llvm::PointerType::get(LLVMTypes::DoubleVectorType, 0);
    LLVMTypes::VoidPointerVectorType = 
        llvm::ArrayType::get(LLVMTypes::VoidPointerType, target.vectorWidth);

    LLVMTrue = llvm::ConstantInt::getTrue(*ctx);
    LLVMFalse = llvm::ConstantInt::getFalse(*ctx);

    std::vector<llvm::Constant *> maskOnes;
    llvm::Constant *onMask = NULL;
    onMask = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), -1,
                                    true /*signed*/); // 0xffffffff

    for (int i = 0; i < target.vectorWidth; ++i)
        maskOnes.push_back(onMask);
    LLVMMaskAllOn = llvm::ConstantVector::get(maskOnes);

    std::vector<llvm::Constant *> maskZeros;
    llvm::Constant *offMask = NULL;
    offMask = llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), 0,
                                     true /*signed*/);

    for (int i = 0; i < target.vectorWidth; ++i)
        maskZeros.push_back(offMask);
    LLVMMaskAllOff = llvm::ConstantVector::get(maskZeros);
}


llvm::ConstantInt *LLVMInt32(int32_t ival) {
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
        assert(LLVMTypes::BoolVectorType->getElementType() == 
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
            assert(LLVMTypes::BoolVectorType->getElementType() == 
                   llvm::Type::getInt1Ty(*g->ctx));
            v = bvec[i] ? LLVMTrue : LLVMFalse;
        }

        vals.push_back(v);
    }
    return llvm::ConstantVector::get(vals);
}


LLVM_TYPE_CONST llvm::ArrayType *
LLVMPointerVectorType(LLVM_TYPE_CONST llvm::Type *t) {
    // NOTE: ArrayType, not VectorType
    return llvm::ArrayType::get(llvm::PointerType::get(t, 0), 
                                g->target.vectorWidth);
}
