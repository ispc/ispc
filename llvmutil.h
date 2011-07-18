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

/** @file llvmutil.h
    @brief Header file with declarations for various LLVM utility stuff
*/

#ifndef ISPC_LLVMUTIL_H
#define ISPC_LLVMUTIL_H 1

#include "ispc.h"
#include <llvm/LLVMContext.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Constants.h>


/** This structure holds pointers to a variety of LLVM types; code
    elsewhere can use them from here, ratherthan needing to make more
    verbose LLVM API calls.
 */ 
struct LLVMTypes {
    static LLVM_TYPE_CONST llvm::Type *VoidType;
    static LLVM_TYPE_CONST llvm::PointerType *VoidPointerType;
    static LLVM_TYPE_CONST llvm::Type *BoolType;
    static LLVM_TYPE_CONST llvm::Type *Int8Type;
    static LLVM_TYPE_CONST llvm::Type *Int16Type;
    static LLVM_TYPE_CONST llvm::Type *Int32Type;
    static LLVM_TYPE_CONST llvm::Type *Int32PointerType;
    static LLVM_TYPE_CONST llvm::Type *Int64Type;
    static LLVM_TYPE_CONST llvm::Type *Int64PointerType;
    static LLVM_TYPE_CONST llvm::Type *FloatType;
    static LLVM_TYPE_CONST llvm::Type *FloatPointerType;
    static LLVM_TYPE_CONST llvm::Type *DoubleType;
    static LLVM_TYPE_CONST llvm::Type *DoublePointerType;

    static LLVM_TYPE_CONST llvm::VectorType *MaskType;
    static LLVM_TYPE_CONST llvm::VectorType *BoolVectorType;
    static LLVM_TYPE_CONST llvm::VectorType *Int1VectorType;
    static LLVM_TYPE_CONST llvm::VectorType *Int32VectorType;
    static LLVM_TYPE_CONST llvm::Type *Int32VectorPointerType;
    static LLVM_TYPE_CONST llvm::VectorType *Int64VectorType;
    static LLVM_TYPE_CONST llvm::Type *Int64VectorPointerType;
    static LLVM_TYPE_CONST llvm::VectorType *FloatVectorType;
    static LLVM_TYPE_CONST llvm::Type *FloatVectorPointerType;
    static LLVM_TYPE_CONST llvm::VectorType *DoubleVectorType;
    static LLVM_TYPE_CONST llvm::Type *DoubleVectorPointerType;
    static LLVM_TYPE_CONST llvm::ArrayType *VoidPointerVectorType;
};

/** These variables hold the corresponding LLVM constant values as a
    convenience to code elsewhere in the system.
 */
extern llvm::Constant *LLVMTrue, *LLVMFalse;

/** This should be called early in initialization to initialize the members
    of LLVMTypes and the LLVMTrue/LLVMFalse constants.  However, it can't
    be called until the compilation target is known.
 */
extern void InitLLVMUtil(llvm::LLVMContext *ctx, Target target);

/** Returns an LLVM i32 constant of the given value */
extern llvm::ConstantInt *LLVMInt32(int32_t i);
/** Returns an LLVM i32 constant of the given value */
extern llvm::ConstantInt *LLVMUInt32(uint32_t i);
/** Returns an LLVM i64 constant of the given value */
extern llvm::ConstantInt *LLVMInt64(int64_t i);
/** Returns an LLVM i64 constant of the given value */
extern llvm::ConstantInt *LLVMUInt64(uint64_t i);
/** Returns an LLVM float constant of the given value */
extern llvm::Constant *LLVMFloat(float f);
/** Returns an LLVM double constant of the given value */
extern llvm::Constant *LLVMDouble(double f);

/** Returns an LLVM boolean vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMBoolVector(bool v);
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
/** Returns an LLVM float vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMFloatVector(float f);
/** Returns an LLVM double vector constant of the given value smeared
    across all elements */
extern llvm::Constant *LLVMDoubleVector(double f);

/** Returns an LLVM boolean vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMBoolVector(const bool *v);
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
/** Returns an LLVM float vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMFloatVector(const float *f);
/** Returns an LLVM double vector based on the given array of values.
    The array should have g->target.vectorWidth elements. */
extern llvm::Constant *LLVMDoubleVector(const double *f);

/** LLVM constant value representing an 'all on' SIMD lane mask */
extern llvm::Constant *LLVMMaskAllOn;
/** LLVM constant value representing an 'all off' SIMD lane mask */
extern llvm::Constant *LLVMMaskAllOff;

/** Given an LLVM type, returns the corresponding type for a vector of
    pointers to that type.  (In practice, an array of pointers, since LLVM
    prohibits vectors of pointers.
 */
extern LLVM_TYPE_CONST llvm::ArrayType *LLVMPointerVectorType(LLVM_TYPE_CONST llvm::Type *t);

#endif // ISPC_LLVMUTIL_H
