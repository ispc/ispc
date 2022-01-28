/*
  Copyright (c) 2010-2022, Intel Corporation
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

/** @file llvmutil.h
    @brief Header file with declarations for various LLVM utility stuff
*/

#pragma once

#include "ispc.h"
#include "ispc_version.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>

// In the transition to Opaque Pointers getElementType() was deprecated, getPointerElementType() will live a little
// longer. But we need another solution eventually. Issue #2245 was filed to track this.
#if ISPC_LLVM_VERSION >= ISPC_LLVM_14_0
#define PTR_ELT_TYPE getPointerElementType
#else
#define PTR_ELT_TYPE getElementType
#endif

#define PTYPE(p) (llvm::cast<llvm::PointerType>((p)->getType()->getScalarType())->PTR_ELT_TYPE())

namespace llvm {
class PHINode;
class InsertElementInst;
} // namespace llvm

#if ISPC_LLVM_VERSION >= ISPC_LLVM_11_0
#define LLVMVECTOR llvm::FixedVectorType
#else
#define LLVMVECTOR llvm::VectorType
#endif

namespace ispc {

/** This structure holds pointers to a variety of LLVM types; code
    elsewhere can use them from here, ratherthan needing to make more
    verbose LLVM API calls.
 */
struct LLVMTypes {
    static llvm::Type *VoidType;
    static llvm::PointerType *VoidPointerType;
    static llvm::Type *PointerIntType;
    static llvm::Type *BoolType;
    static llvm::Type *BoolStorageType;

    static llvm::Type *Int8Type;
    static llvm::Type *Int16Type;
    static llvm::Type *Int32Type;
    static llvm::Type *Int64Type;
    static llvm::Type *Float16Type;
    static llvm::Type *FloatType;
    static llvm::Type *DoubleType;

    static llvm::Type *Int8PointerType;
    static llvm::Type *Int16PointerType;
    static llvm::Type *Int32PointerType;
    static llvm::Type *Int64PointerType;
    static llvm::Type *Float16PointerType;
    static llvm::Type *FloatPointerType;
    static llvm::Type *DoublePointerType;

    static llvm::VectorType *MaskType;

    static llvm::VectorType *BoolVectorType;
    static llvm::VectorType *BoolVectorStorageType;
    static llvm::VectorType *Int1VectorType;
    static llvm::VectorType *Int8VectorType;
    static llvm::VectorType *Int16VectorType;
    static llvm::VectorType *Int32VectorType;
    static llvm::VectorType *Int64VectorType;
    static llvm::VectorType *Float16VectorType;
    static llvm::VectorType *FloatVectorType;
    static llvm::VectorType *DoubleVectorType;

    static llvm::Type *Int8VectorPointerType;
    static llvm::Type *Int16VectorPointerType;
    static llvm::Type *Int32VectorPointerType;
    static llvm::Type *Int64VectorPointerType;
    static llvm::Type *Float16VectorPointerType;
    static llvm::Type *FloatVectorPointerType;
    static llvm::Type *DoubleVectorPointerType;

    static llvm::VectorType *VoidPointerVectorType;
};

/** These variables hold the corresponding LLVM constant values as a
    convenience to code elsewhere in the system.
 */
extern llvm::Constant *LLVMTrue, *LLVMFalse, *LLVMTrueInStorage, *LLVMFalseInStorage;

/** This should be called early in initialization to initialize the members
    of LLVMTypes and the LLVMTrue/LLVMFalse constants.  However, it can't
    be called until the compilation target is known.
 */
class Target;
extern void InitLLVMUtil(llvm::LLVMContext *ctx, Target &target);

/** Returns an LLVM i8 constant of the given value */
extern llvm::ConstantInt *LLVMInt8(int8_t i);
/** Returns an LLVM i8 constant of the given value */
extern llvm::ConstantInt *LLVMUInt8(uint8_t i);
/** Returns an LLVM i16 constant of the given value */
extern llvm::ConstantInt *LLVMInt16(int16_t i);
/** Returns an LLVM i16 constant of the given value */
extern llvm::ConstantInt *LLVMUInt16(uint16_t i);
/** Returns an LLVM i32 constant of the given value */
extern llvm::ConstantInt *LLVMInt32(int32_t i);
/** Returns an LLVM i32 constant of the given value */
extern llvm::ConstantInt *LLVMUInt32(uint32_t i);
/** Returns an LLVM i64 constant of the given value */
extern llvm::ConstantInt *LLVMInt64(int64_t i);
/** Returns an LLVM i64 constant of the given value */
extern llvm::ConstantInt *LLVMUInt64(uint64_t i);
/** Returns an LLVM half constant of the given value */
extern llvm::Constant *LLVMFloat16(llvm::APFloat f);
/** Returns an LLVM float constant of the given value */
extern llvm::Constant *LLVMFloat(llvm::APFloat f);
/** Returns an LLVM double constant of the given value */
extern llvm::Constant *LLVMDouble(llvm::APFloat f);

/** Returns an LLVM boolean vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMBoolVector(bool v);

/** Returns an LLVM boolean vector constant of the given value smeared
    across all elements with bool represented as storage type(i8)*/
extern llvm::Constant *LLVMBoolVectorInStorage(bool v);

/** Returns an LLVM i8 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMInt8Vector(int8_t i);
/** Returns an LLVM i8 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMUInt8Vector(uint8_t i);

/** Returns an LLVM i16 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMInt16Vector(int16_t i);
/** Returns an LLVM i16 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMUInt16Vector(uint16_t i);

/** Returns an LLVM i32 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMInt32Vector(int32_t i);
/** Returns an LLVM i32 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMUInt32Vector(uint32_t i);

/** Returns an LLVM i64 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMInt64Vector(int64_t i);
/** Returns an LLVM i64 vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMUInt64Vector(uint64_t i);

/** Returns an LLVM half vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMFloat16Vector(llvm::APFloat f);
/** Returns an LLVM float vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMFloatVector(llvm::APFloat f);
/** Returns an LLVM double vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMDoubleVector(llvm::APFloat f);

/** Returns a constant integer or vector (according to the given type) of
    the given signed integer value. */
extern llvm::Constant *LLVMIntAsType(int64_t, llvm::Type *t);

/** Returns a constant integer or vector (according to the given type) of
    the given unsigned integer value. */
extern llvm::Constant *LLVMUIntAsType(uint64_t, llvm::Type *t);

/** Returns an LLVM boolean vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMBoolVector(const bool *v);

/** Returns an LLVM boolean vector based on the given array of values
    with bool represented as storage type(i8).
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMBoolVectorInStorage(const bool *v);

/** Returns an LLVM i8 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMInt8Vector(const int8_t *i);
/** Returns an LLVM i8 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMUInt8Vector(const uint8_t *i);

/** Returns an LLVM i16 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMInt16Vector(const int16_t *i);
/** Returns an LLVM i16 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMUInt16Vector(const uint16_t *i);

/** Returns an LLVM i32 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMInt32Vector(const int32_t *i);
/** Returns an LLVM i32 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMUInt32Vector(const uint32_t *i);

/** Returns an LLVM i64 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMInt64Vector(const int64_t *i);
/** Returns an LLVM i64 vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMUInt64Vector(const uint64_t *i);

/** Returns an LLVM half vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMFloat16Vector(const std::vector<llvm::APFloat> &f);
/** Returns an LLVM float vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMFloatVector(const std::vector<llvm::APFloat> &f);
/** Returns an LLVM double vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMDoubleVector(const std::vector<llvm::APFloat> &f);

/** LLVM constant value representing an 'all on' SIMD lane mask */
extern llvm::Constant *LLVMMaskAllOn;
/** LLVM constant value representing an 'all off' SIMD lane mask */
extern llvm::Constant *LLVMMaskAllOff;

/** Tests to see if all of the elements of the vector in the 'v' parameter
    are equal.  Like lValuesAreEqual(), this is a conservative test and may
    return false for arrays where the values are actually all equal.  */
extern bool LLVMVectorValuesAllEqual(llvm::Value *v, llvm::Value **splat = NULL);

/** Tests to see if OR is actually an ADD.  */
extern bool IsOrEquivalentToAdd(llvm::Value *op);

/** Given vector of integer-typed values, this function returns true if it
    can determine that the elements of the vector have a step of 'stride'
    between their values and false otherwise.  This function tries to
    handle as many possibilities as possible, including things like all
    elements equal to some non-constant value plus an integer offset, etc.
    Needless to say (the halting problem and all that), it may return false
    for some vectors that are in fact linear.
    */
extern bool LLVMVectorIsLinear(llvm::Value *v, int stride);

/** Given a vector-typed value v, if the vector is a vector with constant
    element values, this function extracts those element values into the
    ret[] array and returns the number of elements (i.e. the vector type's
    width) in *nElts.  It returns true if successful and false if the given
    vector is not in fact a vector of constants. */
extern bool LLVMExtractVectorInts(llvm::Value *v, int64_t ret[], int *nElts);

/** This function takes chains of InsertElement instructions along the
    lines of:

    %v0 = insertelement undef, value_0, i32 index_0
    %v1 = insertelement %v1,   value_1, i32 index_1
    ...
    %vn = insertelement %vn-1, value_n-1, i32 index_n-1

    and initializes the provided elements array such that the i'th
    llvm::Value * in the array is the element that was inserted into the
    i'th element of the vector.

    When the chain of insertelement instruction comes to an end, the only
    base case that this function handles is the initial value being a
    constant vector.  For anything more complex (e.g. some other arbitrary
    value, it doesn't try to extract element values into the returned
    array.

    This also handles one of two common broadcast patterns:
    1.   %broadcast_init.0 = insertelement <4 x i32> undef, i32 %val, i32 0
         %broadcast.1 = shufflevector <4 x i32> %smear.0, <4 x i32> undef,
                                                  <4 x i32> zeroinitializer
    2.   %gep_ptr2int_broadcast_init = insertelement <8 x i64> undef, i64 %gep_ptr2int, i32 0
         %0 = add <8 x i64> %gep_ptr2int_broadcast_init,
                  <i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>
         %gep_offset = shufflevector <8 x i64> %0, <8 x i64> undef, <8 x i32> zeroinitializer
    Function returns:
    Compare all elements and return one of them if all are equal, otherwise NULL.
    If searchFirstUndef argument is true, look for the vector with the first not-undef element, like:
         <i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>
    If compare argument is false, don't do compare and return first element instead.
    If undef argument is true, ignore undef elements (but all undef yields NULL anyway).

 */
extern llvm::Value *LLVMFlattenInsertChain(llvm::Value *inst, int vectorWidth, bool compare = true, bool undef = true,
                                           bool searchFirstUndef = false);

/** This is a utility routine for debugging that dumps out the given LLVM
    value as well as (recursively) all of the other values that it depends
    on. */
#ifndef ISPC_NO_DUMPS
extern void LLVMDumpValue(llvm::Value *v);
#endif

/** Given a vector-typed value, this function returns the value of its
    first element.  Rather than just doing the straightforward thing of
    using a single extractelement instruction to do this, this function
    tries to rewrite the computation for the first element in scalar form;
    this is generally more efficient than computing the entire vector's
    worth of values just to extract the first element, in cases where only
    the first element's value is needed.
  */
extern llvm::Value *LLVMExtractFirstVectorElement(llvm::Value *v);

/** This function takes two vectors, expected to be the same length, and
    returns a new vector of twice the length that represents concatenating
    the two of them. */
extern llvm::Value *LLVMConcatVectors(llvm::Value *v1, llvm::Value *v2, llvm::Instruction *insertBefore);

/** This is a utility function for vector shuffling; it takes two vectors
    v1 and v2, and a compile-time constant set of integer permutations in
    shuf[] and returns a new vector of length shufSize that represents the
    corresponding shufflevector operation. */
extern llvm::Value *LLVMShuffleVectors(llvm::Value *v1, llvm::Value *v2, int32_t shuf[], int shufSize,
                                       llvm::Instruction *insertBefore);

#ifdef ISPC_XE_ENABLED
/** This is utility function to determine memory in which pointer was created.
    For now we use only 3 values:
    Global is used for pointers passed to kernel externally
    Private is used for pointers created locally with alloca
    Generic is currently used to identify Global variables
*/
extern AddressSpace GetAddressSpace(llvm::Value *v);
#endif
} // namespace ispc
