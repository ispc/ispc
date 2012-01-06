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

/** @file type.h
    @brief File with declarations for classes related to type representation
*/

#ifndef ISPC_TYPE_H
#define ISPC_TYPE_H 1

#include "ispc.h"
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>

class ConstExpr;
class StructType;

/** @brief Interface class that defines the type abstraction.

    Abstract base class that defines the interface that must be implemented
    for all types in the language.
 */
class Type {
public:
    /** Returns true if the underlying type is boolean.  In other words,
        this is true for individual bools and for short-vectors with
        underlying bool type, but not for arrays of bools. */
    virtual bool IsBoolType() const = 0;

    /** Returns true if the underlying type is float or double.  In other
        words, this is true for individual floats/doubles and for
        short-vectors of them, but not for arrays of them. */
    virtual bool IsFloatType() const = 0;

    /** Returns true if the underlying type is an integer type.  In other
        words, this is true for individual integers and for short-vectors
        of integer types, but not for arrays of integer types. */
    virtual bool IsIntType() const = 0;

    /** Returns true if the underlying type is unsigned.  In other words,
        this is true for unsigned integers and short vectors of unsigned
        integer types. */
    virtual bool IsUnsignedType() const = 0;

    /** Returns true if this type is 'const'-qualified. */
    virtual bool IsConstType() const = 0;
    
    /** Returns true if the underlying type is a float or integer type. */
    bool IsNumericType() const { return IsFloatType() || IsIntType(); }

    /** Types may have uniform, varying, or not-yet-determined variability;
        this enumerant is used by Type implementations to record their
        variability. */
    enum Variability {
        Uniform,
        Varying,
        Unbound
    };

    /** Returns the variability of the type. */
    virtual Variability GetVariability() const = 0;

    /** Returns true if the underlying type is uniform */
    bool IsUniformType() const { return GetVariability() == Uniform; }

    /** Returns true if the underlying type is varying */
    bool IsVaryingType() const { return GetVariability() == Varying; }

    /** Returns true if the underlying type's uniform/varying-ness is
        unbound. */
    bool HasUnboundVariability() const { return GetVariability() == Unbound; }

    /* Returns a type wherein any elements of the original type and
       contained types that have unbound variability have their variability
       set to the given variability. */
    virtual const Type *ResolveUnboundVariability(Variability v) const = 0;

    /** Return a "uniform" instance of this type.  If the type is already
        uniform, its "this" pointer will be returned. */
    virtual const Type *GetAsUniformType() const = 0;

    /** Return a "varying" instance of this type.  If the type is already
        varying, its "this" pointer will be returned. */
    virtual const Type *GetAsVaryingType() const = 0;

    /** Get an instance of the type with unbound variability. */
    virtual const Type *GetAsUnboundVariabilityType() const = 0;

    /** If this is a signed integer type, return the unsigned version of
        the type.  Otherwise, return the original type. */
    virtual const Type *GetAsUnsignedType() const;

    /** Returns the basic root type of the given type.  For example, for an
        array or short-vector, this returns the element type.  For a struct
        or atomic type, it returns itself. */
    virtual const Type *GetBaseType() const = 0;

    /** If this is a reference type, returns the type it is referring to.
        For all other types, just returns its own type. */
    virtual const Type *GetReferenceTarget() const;

    /** Return a new type representing the current type laid out in
        width-wide SOA (structure of arrays) format. */
    virtual const Type *GetSOAType(int width) const = 0;

    /** Get a const version of this type.  If it's already const, then the old 
        Type pointer is returned. */
    virtual const Type *GetAsConstType() const = 0;

    /** Get a non-const version of this type.  If it's already not const,
        then the old Type pointer is returned. */
    virtual const Type *GetAsNonConstType() const = 0;

    /** Returns a text representation of the type (for example, for use in
        warning and error messages). */
    virtual std::string GetString() const = 0;

    /** Returns a string that represents the mangled type (for use in
        mangling function symbol names for function overloading).  The
        various Types implementations of this method should collectively
        ensure that all of them use mangling schemes that are guaranteed
        not to clash. */
    virtual std::string Mangle() const = 0;

    /** Returns a string that is the declaration of the same type in C
        syntax. */
    virtual std::string GetCDeclaration(const std::string &name) const = 0;

    /** Returns the LLVM type corresponding to this ispc type */
    virtual LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const = 0;

    /** Returns the DIType (LLVM's debugging information structure),
        corresponding to this type. */
    virtual llvm::DIType GetDIType(llvm::DIDescriptor scope) const = 0;

    /** Checks two types for equality.  Returns true if they are exactly
        the same, false otherwise. */
    static bool Equal(const Type *a, const Type *b);

    /** Checks two types for equality.  Returns true if they are exactly
        the same (ignoring const-ness of the type), false otherwise. */
    static bool EqualIgnoringConst(const Type *a, const Type *b);

    /** Given two types, returns the least general Type that is more general
        than both of them.  (i.e. that can represent their values without
        any loss of data.)  If there is no such Type, return NULL.

        @param type0        First of the two types
        @param type1        Second of the two types
        @param pos          Source file position where the general type is
                            needed.
        @param reason       String describing the context of why the general
                            type is needed (e.g. "+ operator").
        @param forceVarying If \c true, then make sure that the returned 
                            type is "varying".
        @param vecSize      The vector size of the returned type.  If non-zero,
                            the returned type will be a VectorType of the
                            more general type with given length.  If zero,
                            this parameter has no effect.
        @return             The more general type, based on the provided parameters.

        @todo the vecSize and forceVarying parts of this should probably be
        factored out and done separately in the cases when needed.
        
    */
    static const Type *MoreGeneralType(const Type *type0, const Type *type1,
                                       SourcePos pos, const char *reason,
                                       bool forceVarying = false, int vecSize = 0);
};


/** @brief AtomicType represents basic types like floats, ints, etc.  

    AtomicTypes can be either uniform or varying.  Unique instances of all
    of the possible <tt>AtomicType</tt>s are available in the static members
    like AtomicType::UniformInt32.  It is thus possible to compare
    AtomicTypes for equality with simple pointer equality tests; this is
    not true for the other Type implementations.
 */
class AtomicType : public Type {
public:
    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    /** For AtomicTypes, the base type is just the same as the AtomicType
        itself. */
    const AtomicType *GetBaseType() const;
    const AtomicType *GetAsUniformType() const;
    const AtomicType *GetAsVaryingType() const;
    const AtomicType *GetAsUnboundVariabilityType() const;
    const AtomicType *ResolveUnboundVariability(Variability v) const;
    const AtomicType *GetAsUnsignedType() const;
    const Type *GetSOAType(int width) const;
    const AtomicType *GetAsConstType() const;
    const AtomicType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

    /** This enumerator records the basic types that AtomicTypes can be 
        built from.  */
    enum BasicType {
        TYPE_VOID,
        TYPE_BOOL,
        TYPE_INT8,
        TYPE_UINT8,
        TYPE_INT16,
        TYPE_UINT16,
        TYPE_INT32,
        TYPE_UINT32,
        TYPE_FLOAT,
        TYPE_INT64,
        TYPE_UINT64,
        TYPE_DOUBLE,
        NUM_BASIC_TYPES
    };

    const BasicType basicType;

    static const AtomicType *UniformBool, *VaryingBool, *UnboundBool;
    static const AtomicType *UniformInt8, *VaryingInt8, *UnboundInt8;
    static const AtomicType *UniformInt16, *VaryingInt16, *UnboundInt16;
    static const AtomicType *UniformInt32, *VaryingInt32, *UnboundInt32;
    static const AtomicType *UniformUInt8, *VaryingUInt8, *UnboundUInt8;
    static const AtomicType *UniformUInt16, *VaryingUInt16, *UnboundUInt16;
    static const AtomicType *UniformUInt32, *VaryingUInt32, *UnboundUInt32;
    static const AtomicType *UniformFloat, *VaryingFloat, *UnboundFloat;
    static const AtomicType *UniformInt64, *VaryingInt64, *UnboundInt64;
    static const AtomicType *UniformUInt64, *VaryingUInt64, *UnboundUInt64;
    static const AtomicType *UniformDouble, *VaryingDouble, *UnboundDouble;
    static const AtomicType *UniformConstBool, *VaryingConstBool, *UnboundConstBool;
    static const AtomicType *UniformConstInt8, *VaryingConstInt8, *UnboundConstInt8;
    static const AtomicType *UniformConstInt16, *VaryingConstInt16, *UnboundConstInt16;
    static const AtomicType *UniformConstInt32, *VaryingConstInt32, *UnboundConstInt32;
    static const AtomicType *UniformConstUInt8, *VaryingConstUInt8, *UnboundConstUInt8;
    static const AtomicType *UniformConstUInt16, *VaryingConstUInt16, *UnboundConstUInt16;
    static const AtomicType *UniformConstUInt32, *VaryingConstUInt32, *UnboundConstUInt32;
    static const AtomicType *UniformConstFloat, *VaryingConstFloat, *UnboundConstFloat;
    static const AtomicType *UniformConstInt64, *VaryingConstInt64, *UnboundConstInt64;
    static const AtomicType *UniformConstUInt64, *VaryingConstUInt64, *UnboundConstUInt64;
    static const AtomicType *UniformConstDouble, *VaryingConstDouble, *UnboundConstDouble;
    static const AtomicType *Void;

    /** This function must be called before any of the above static const
        AtomicType values is used; in practice, we do it early in
        main(). */
    static void Init();

private:
    static const AtomicType *typeTable[NUM_BASIC_TYPES][3][2];
    const Variability variability;
    const bool isConst;
    AtomicType(BasicType basicType, Variability v, bool isConst);
};


/** @brief Type implementation for enumerated types
 */
class EnumType : public Type {
public:
    /** Constructor for anonymous enumerated types */
    EnumType(SourcePos pos);
    /** Constructor for named enumerated types */
    EnumType(const char *name, SourcePos pos);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const EnumType *GetBaseType() const;
    const EnumType *GetAsVaryingType() const;
    const EnumType *GetAsUniformType() const;
    const EnumType *GetAsUnboundVariabilityType() const;
    const EnumType *ResolveUnboundVariability(Variability v) const;
    const Type *GetSOAType(int width) const;
    const EnumType *GetAsConstType() const;
    const EnumType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

    /** Provides the enumerators defined in the enum definition. */
    void SetEnumerators(const std::vector<Symbol *> &enumerators);
    /** Returns the total number of enuemrators in this enum type. */
    int GetEnumeratorCount() const;
    /** Returns the symbol for the given enumerator number. */
    const Symbol *GetEnumerator(int i) const;

    const SourcePos pos;

private:
    const std::string name;
    Variability variability;
    bool isConst;
    std::vector<Symbol *> enumerators;
};


/** @brief Type implementation for pointers to other types
 */
class PointerType : public Type {
public:
    PointerType(const Type *t, Variability v, bool isConst);

    /** Helper method to return a uniform pointer to the given type. */
    static PointerType *GetUniform(const Type *t);
    /** Helper method to return a varying pointer to the given type. */
    static PointerType *GetVarying(const Type *t);

    /** Returns true if the given type is a void * type. */
    static bool IsVoidPointer(const Type *t);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const Type *GetBaseType() const;
    const PointerType *GetAsVaryingType() const;
    const PointerType *GetAsUniformType() const;
    const PointerType *GetAsUnboundVariabilityType() const;
    const PointerType *ResolveUnboundVariability(Variability v) const;
    const Type *GetSOAType(int width) const;
    const PointerType *GetAsConstType() const;
    const PointerType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

    static PointerType *Void;

private:
    const Variability variability;
    const bool isConst;
    const Type *baseType;
};


/** @brief Abstract base class for types that represent collections of
    other types.

    This is a common base class that StructTypes, ArrayTypes, and
    VectorTypes all inherit from.
*/ 
class CollectionType : public Type {
public:
    /** Returns the total number of elements in the collection. */
    virtual int GetElementCount() const = 0;

    /** Returns the type of the element given by index.  (The value of
        index must be between 0 and GetElementCount()-1.
     */
    virtual const Type *GetElementType(int index) const = 0;
};


/** @brief Abstract base class for types that represent sequences

    SequentialType is an abstract base class that adds interface routines
    for types that represent linear sequences of other types (i.e., arrays
    and vectors).
 */
class SequentialType : public CollectionType {
public:
    /** Returns the Type of the elements that the sequence stores; for
        SequentialTypes, all elements have the same type . */
    virtual const Type *GetElementType() const = 0;

    /** SequentialType provides an implementation of this CollectionType
        method, just passing the query on to the GetElementType(void)
        implementation, since all of the elements of a SequentialType have
        the same type.
     */
    const Type *GetElementType(int index) const;
};


/** @brief One-dimensional array type.

    ArrayType represents a one-dimensional array of instances of some other
    type.  (Multi-dimensional arrays are represented by ArrayTypes that in
    turn hold ArrayTypes as their child types.)  
*/
class ArrayType : public SequentialType {
public:
    /** An ArrayType is created by providing the type of the elements that
        it stores, and the SOA width to use in laying out the array in
        memory.

        @param elementType  Type of the array elements
        @param numElements  Total number of elements in the array.  This
                            parameter may be zero, in which case this is an
                            "unsized" array type.  (Arrays of specific size
                            can be converted to unsized arrays to be passed
                            to functions that take array parameters, for
                            example).
     */
    ArrayType(const Type *elementType, int numElements);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const Type *GetBaseType() const;
    const ArrayType *GetAsVaryingType() const;
    const ArrayType *GetAsUniformType() const;
    const ArrayType *GetAsUnboundVariabilityType() const;
    const ArrayType *ResolveUnboundVariability(Variability v) const;

    const ArrayType *GetAsUnsignedType() const;
    const Type *GetSOAType(int width) const;
    const ArrayType *GetAsConstType() const;
    const ArrayType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;
    LLVM_TYPE_CONST llvm::ArrayType *LLVMType(llvm::LLVMContext *ctx) const;

    /** This method returns the total number of elements in the array,
        including all dimensions if this is a multidimensional array. */
    int TotalElementCount() const;

    int GetElementCount() const;
    const Type *GetElementType() const;

    /** Returns a new array of the same child type, but with the given
        length. */
    virtual ArrayType *GetSizedArray(int length) const;

    /** If the given type is a (possibly multi-dimensional) array type and
        the initializer expression is an expression list, set the size of
        any array dimensions that are unsized according to the number of
        elements in the corresponding sectoin of the initializer
        expression.
     */ 
    static const Type *SizeUnsizedArrays(const Type *type, Expr *initExpr);

private:
    friend class SOAArrayType;
    /** Type of the elements of the array. */
    const Type * const child;
    /** Number of elements in the array. */
    const int numElements;
};


/** @brief "Structure of arrays" array type.

    This type represents an array with elements of a structure type,
    "SOA-ized" to some width w.  This corresponds to replicating the struct
    element types w times and then having an array of size/w of these
    widened structs.  This memory layout often makes it possible to access
    data with regular vector loads, rather than gathers that are needed
    with "AOS" (array of structures) layout.

    @todo Native support for SOA stuff is still a work in progres...
 */
class SOAArrayType : public ArrayType {
public:
    /**
       SOAType constructor.

       @param elementType  Type of the array elements.  Must be a StructType.
       @param numElements  Total number of elements in the array.  This
                           parameter may be zero, in which case this is an
                           "unsized" array type.  (Arrays of specific size
                           can be converted to unsized arrays to be passed
                           to functions that take array parameters, for
                           example).
       @param soaWidth     If non-zero, this gives the SOA width to use in
                           laying out the array data in memory.  (This value
                           must be a power of two).  For example, if the
                           array's element type is: 
                           <tt>struct { uniform float x, y, z; }</tt>,
                           the SOA width is four, and the number of elements
                           is 12, then the array will be laid out in memory
                           as xxxxyyyyzzzzxxxxyyyyzzzzxxxxyyyyzzzz.
    */
    SOAArrayType(const StructType *elementType, int numElements, 
                 int soaWidth);

    const SOAArrayType *GetAsVaryingType() const;
    const SOAArrayType *GetAsUniformType() const;
    const SOAArrayType *GetAsUnboundVariabilityType() const;
    const SOAArrayType *ResolveUnboundVariability(Variability v) const;

    const Type *GetSOAType(int width) const;
    const SOAArrayType *GetAsConstType() const;
    const SOAArrayType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    int TotalElementCount() const;

    LLVM_TYPE_CONST llvm::ArrayType *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

    SOAArrayType *GetSizedArray(int size) const;

private:
    /** This member variable records the rate at which the structure
        elements are replicated. */
    const int soaWidth;

    /** Returns a regular ArrayType with the struct type's elements widened
        out and with correspondingly fewer array elements. */
    const ArrayType *soaType() const;
};


/** @brief A (short) vector of atomic types.

    VectorType is used to represent a fixed-size array of elements of an
    AtomicType.  Vectors are similar to arrays in that they support
    indexing of the elements, but have two key differences.  First, all
    arithmetic and logical operations that are value for the element type
    can be performed on corresponding VectorTypes (as long as the two
    VectorTypes have the same size). Second, VectorTypes of uniform
    elements are laid out in memory aligned to the target's vector size;
    this allows them to be packed 'horizontally' into vector registers.
 */
class VectorType : public SequentialType {
public:
    VectorType(const AtomicType *base, int size);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const Type *GetBaseType() const;
    const VectorType *GetAsVaryingType() const;
    const VectorType *GetAsUniformType() const;
    const VectorType *GetAsUnboundVariabilityType() const;
    const VectorType *ResolveUnboundVariability(Variability v) const;

    const Type *GetSOAType(int width) const;
    const VectorType *GetAsConstType() const;
    const VectorType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

    int GetElementCount() const;
    const AtomicType *GetElementType() const;

private:
    /** Base type that the vector holds elements of */
    const AtomicType * const base;
    /** Number of elements in the vector */
    const int numElements;

    /** Returns the number of elements stored in memory for the vector.
        For uniform vectors, this is rounded up so that the number of
        elements evenly divides the target's native vector width. */
    int getVectorMemoryCount() const;
};


/** @brief Representation of a structure holding a number of members.
 */
class StructType : public CollectionType {
public:
    StructType(const std::string &name, const std::vector<const Type *> &elts, 
               const std::vector<std::string> &eltNames, 
               const std::vector<SourcePos> &eltPositions, bool isConst, 
               Variability variability, SourcePos pos);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const Type *GetBaseType() const;
    const StructType *GetAsVaryingType() const;
    const StructType *GetAsUniformType() const;
    const StructType *GetAsUnboundVariabilityType() const;
    const StructType *ResolveUnboundVariability(Variability v) const;

    const Type *GetSOAType(int width) const;
    const StructType *GetAsConstType() const;
    const StructType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

    /** Returns the type of the structure element with the given name (if any).
        Returns NULL if there is no such named element. */
    const Type *GetElementType(const std::string &name) const;

    /** Returns the type of the i'th structure element.  The value of \c i must
        be between 0 and NumElements()-1. */
    const Type *GetElementType(int i) const;

    /** Returns which structure element number (starting from zero) that
        has the given name.  If there is no such element, return -1. */
    int GetElementNumber(const std::string &name) const;

    /** Returns the name of the i'th element of the structure. */
    const std::string GetElementName(int i) const { return elementNames[i]; }
    
    /** Returns the total number of elements in the structure. */
    int GetElementCount() const { return int(elementTypes.size()); }

    /** Returns the name of the structure type.  (e.g. struct Foo -> "Foo".) */
    const std::string &GetStructName() const { return name; }

private:
    const std::string name;
    /** The types of the struct elements.  Note that we store these with
        uniform/varying exactly as they were declared in the source file.
        (In other words, even if this struct has a varying qualifier and
        thus all of its members are going to be widened out to be varying,
        we still store any members that were declared as uniform as uniform
        types in the elementTypes array, converting them to varying as
        needed in the implementation.)  This is so that if we later need to
        make a uniform version of the struct, we've maintained the original
        information about the member types.
     */
    const std::vector<const Type *> elementTypes;
    const std::vector<std::string> elementNames;
    /** Source file position at which each structure element declaration
        appeared. */
    const std::vector<SourcePos> elementPositions;
    const Variability variability;
    const bool isConst;
    const SourcePos pos;
};


/** @brief Type representing a reference to another (non-reference) type.
 */
class ReferenceType : public Type {
public:
    ReferenceType(const Type *targetType);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const Type *GetBaseType() const;
    const Type *GetReferenceTarget() const;
    const ReferenceType *GetAsVaryingType() const;
    const ReferenceType *GetAsUniformType() const;
    const ReferenceType *GetAsUnboundVariabilityType() const;
    const ReferenceType *ResolveUnboundVariability(Variability v) const;

    const Type *GetSOAType(int width) const;
    const ReferenceType *GetAsConstType() const;
    const ReferenceType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

private:
    const Type * const targetType;
};


/** @brief Type representing a function (return type + argument types)

    FunctionType encapsulates the information related to a function's type,
    including the return type and the types of the arguments.

    @todo This class has a fair number of methods inherited from Type that
    don't make sense here (e.g. IsUniformType(), GetBaseType(), LLVMType(), etc.
    Would be nice to refactor the inheritance hierarchy to move most of
    those interface methods to a sub-class of Type, which in turn all of
    the other Type implementations inherit from.
 */
class FunctionType : public Type {
public:
    FunctionType(const Type *returnType, 
                 const std::vector<const Type *> &argTypes, SourcePos pos);
    FunctionType(const Type *returnType, 
                 const std::vector<const Type *> &argTypes,
                 const std::vector<std::string> &argNames,
                 const std::vector<ConstExpr *> &argDefaults,
                 const std::vector<SourcePos> &argPos,
                 bool isTask, bool isExported, bool isExternC);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const Type *GetBaseType() const;
    const Type *GetAsVaryingType() const;
    const Type *GetAsUniformType() const;
    const Type *GetAsUnboundVariabilityType() const;
    const FunctionType *ResolveUnboundVariability(Variability v) const;

    const Type *GetSOAType(int width) const;
    const Type *GetAsConstType() const;
    const Type *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &fname) const;

    LLVM_TYPE_CONST llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;
    llvm::DIType GetDIType(llvm::DIDescriptor scope) const;

    const Type *GetReturnType() const { return returnType; }

    /** This method returns the LLVM FunctionType that corresponds to this
        function type.  The \c includeMask parameter indicates whether the
        llvm::FunctionType should have a mask as the last argument in its
        function signature. */
    LLVM_TYPE_CONST llvm::FunctionType *LLVMFunctionType(llvm::LLVMContext *ctx, 
                                                         bool includeMask = false) const;

    int GetNumParameters() const { return (int)paramTypes.size(); }
    const Type *GetParameterType(int i) const;
    ConstExpr * GetParameterDefault(int i) const;
    const SourcePos &GetParameterSourcePos(int i) const;
    const std::string &GetParameterName(int i) const;

    /** This value is true if the function had a 'task' qualifier in the
        source program. */
    const bool isTask;

    /** This value is true if the function had a 'export' qualifier in the
        source program. */
    const bool isExported;

    /** This value is true if the function was declared as an 'extern "C"'
        function in the source program. */
    const bool isExternC;

private:
    const Type * const returnType;

    // The following four vectors should all have the same length (which is
    // in turn the length returned by GetNumParameters()).
    const std::vector<const Type *> paramTypes;
    const std::vector<std::string> paramNames;
    /** Default values of the function's arguments.  For arguments without
        default values provided, NULL is stored. */
    mutable std::vector<ConstExpr *> paramDefaults;
    /** The names provided (if any) with the function arguments in the
        function's signature.  These should only be used for error messages
        and the like and so not affect testing function types for equality,
        etc. */
    const std::vector<SourcePos> paramPositions;
};

#endif // ISPC_TYPE_H
