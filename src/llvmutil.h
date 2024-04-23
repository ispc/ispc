/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file llvmutil.h
    @brief Header file with declarations for various LLVM utility stuff
*/

#pragma once

#include "ispc.h"
#include "ispc_version.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
#include <llvm/Support/ModRef.h>
#endif

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

#define LLVMVECTOR llvm::FixedVectorType

namespace ispc {

enum class MaskStatus { all_on, all_off, mixed, unknown };

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

/** Returns a zero constant half/float/double or vector (according to the given type). */
extern llvm::Constant *LLVMFPZeroAsType(llvm::Type *type);

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
extern bool LLVMVectorValuesAllEqual(llvm::Value *v, llvm::Value **splat = nullptr);

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
    Compare all elements and return one of them if all are equal, otherwise nullptr.
    If searchFirstUndef argument is true, look for the vector with the first not-undef element, like:
         <i64 4, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef, i64 undef>
    If compare argument is false, don't do compare and return first element instead.
    If undef argument is true, ignore undef elements (but all undef yields nullptr anyway).

 */
extern llvm::Value *LLVMFlattenInsertChain(llvm::Value *inst, int vectorWidth, bool compare = true, bool undef = true,
                                           bool searchFirstUndef = false);

/** This is a utility routine for debugging that dumps out the given LLVM
    value as well as (recursively) all of the other values that it depends
    on. */
extern void LLVMDumpValue(llvm::Value *v);

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

/** This utility routine copies the metadata (if any) attached to the
    'from' instruction in the IR to the 'to' instruction.

    For flexibility, this function takes an llvm::Value rather than an
    llvm::Instruction for the 'to' parameter; at some places in the code
    below, we sometimes use a llvm::Value to start out storing a value and
    then later store instructions.  If a llvm::Value is passed to this, the
    routine just returns without doing anything; if it is in fact an
    LLVM::Instruction, then the metadata can be copied to it. */
extern void LLVMCopyMetadata(llvm::Value *vto, const llvm::Instruction *from);

/** We have a protocol with the front-end LLVM IR code generation process
    that allows us to encode the source file position that corresponds with
    instructions.  (For example, this allows us to issue performance
    warnings related to things like scatter and gather after optimization
    has been performed, so that we aren't warning about scatters and
    gathers that have been improved to stores and loads by optimization
    passes.)  Note that this is slightly redundant with the source file
    position encoding generated for debugging symbols, though we don't
    always generate debugging information but we do always generate this
    position data.

    This function finds the SourcePos that the metadata in the instruction
    (if present) corresponds to.  See the implementation of
    FunctionEmitContext::addGSMetadata(), which encodes the source position during
    code generation.

    @param inst   Instruction to try to find the source position of
    @param pos    Output variable in which to store the position
    @returns      True if source file position metadata was present and *pos
                  has been set.  False otherwise.*/
extern bool LLVMGetSourcePosFromMetadata(const llvm::Instruction *inst, SourcePos *pos);

/** Given an llvm::Value, return true if we can determine that it's an
    undefined value.  This only makes a weak attempt at chasing this down,
    only detecting flat-out undef values, and bitcasts of undef values.

    @todo Is it worth working harder to find more of these?  It starts to
    get tricky, since having an undef operand doesn't necessarily mean that
    the result will be undefined.  (And for that matter, is there an LLVM
    call that will do this for us?)
 */
extern bool LLVMIsValueUndef(llvm::Value *value);

/** Below are helper functions to construct LLVM instructions. */
extern llvm::CallInst *LLVMCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = nullptr);

extern llvm::CallInst *LLVMCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    const llvm::Twine &name, llvm::Instruction *insertBefore = nullptr);

extern llvm::CallInst *LLVMCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    llvm::Value *arg3, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = nullptr);

extern llvm::CallInst *LLVMCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    llvm::Value *arg3, llvm::Value *arg4, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = nullptr);

extern llvm::CallInst *LLVMCallInst(llvm::Function *func, llvm::Value *arg0, llvm::Value *arg1, llvm::Value *arg2,
                                    llvm::Value *arg3, llvm::Value *arg4, llvm::Value *arg5, const llvm::Twine &name,
                                    llvm::Instruction *insertBefore = nullptr);

extern llvm::GetElementPtrInst *LLVMGEPInst(llvm::Value *ptr, llvm::Type *ptrElType, llvm::Value *offset,
                                            const char *name, llvm::Instruction *insertBefore);

/** Mask-related helpers */

/** Given an llvm::Value represinting a vector mask, see if the value is a
    constant.  If so, return true and set *bits to be the integer mask
    found by taking the high bits of the mask values in turn and
    concatenating them into a single integer.  In other words, given the
    4-wide mask: < 0xffffffff, 0, 0, 0xffffffff >, we have 0b1001 = 9.
 */
extern bool GetMaskFromValue(llvm::Value *factor, uint64_t *mask);

/** Determines if the given mask value is all on, all off, mixed, or
    unknown at compile time.
*/
extern MaskStatus GetMaskStatusFromValue(llvm::Value *mask, int vecWidth = -1);

/** Add uwtable attribute for function, windows specific.
 */
extern void AddUWTableFuncAttr(llvm::Function *fn);

#ifdef ISPC_XE_ENABLED
/** This is utility function to determine memory in which pointer was created.
    For now we use only 3 values:
    Global is used for pointers passed to kernel externally
    Private is used for pointers created locally with alloca
    Generic is currently used to identify Global variables
*/
extern AddressSpace GetAddressSpace(llvm::Value *v);

#if ISPC_LLVM_VERSION >= ISPC_LLVM_16_0
/** Fix function attribute by removing input function attr and adding memory effect instead.
    https://reviews.llvm.org/D135780
*/
extern void FixFunctionAttribute(llvm::Function &Fn, llvm::Attribute::AttrKind attr, llvm::MemoryEffects memEf);
#endif
#endif
} // namespace ispc
