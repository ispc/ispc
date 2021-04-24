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

/** @file type.h
    @brief File with declarations for classes related to type representation
*/

#pragma once

#include "ispc.h"
#include "util.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

namespace ispc {

class ConstExpr;
class StructType;

/** Types may have uniform, varying, SOA, or unbound variability; this
    struct is used by Type implementations to record their variability.
*/
struct Variability {
    enum VarType { Unbound, Uniform, Varying, SOA };

    Variability(VarType t = Unbound, int w = 0) : type(t), soaWidth(w) {}

    bool operator==(const Variability &v) const { return v.type == type && v.soaWidth == soaWidth; }
    bool operator!=(const Variability &v) const { return v.type != type || v.soaWidth != soaWidth; }

    bool operator==(const VarType &t) const { return type == t; }
    bool operator!=(const VarType &t) const { return type != t; }

    std::string GetString() const;
    std::string MangleString() const;

    VarType type;
    int soaWidth;
};

/** Enumerant that records each of the types that inherit from the Type
    baseclass. */
enum TypeId {
    ATOMIC_TYPE,           // 0
    ENUM_TYPE,             // 1
    POINTER_TYPE,          // 2
    ARRAY_TYPE,            // 3
    VECTOR_TYPE,           // 4
    STRUCT_TYPE,           // 5
    UNDEFINED_STRUCT_TYPE, // 6
    REFERENCE_TYPE,        // 7
    FUNCTION_TYPE          // 8
};

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

    /** Returns true if the underlying type is either a pointer type */
    bool IsPointerType() const;

    /** Returns true if the underlying type is a array type */
    bool IsArrayType() const;

    /** Returns true if the underlying type is a array type */
    bool IsReferenceType() const;

    /** Returns true if the underlying type is either a pointer or an array */
    bool IsVoidType() const;

    /** Returns true if this type is 'const'-qualified. */
    virtual bool IsConstType() const = 0;

    /** Returns true if the underlying type is a float or integer type. */
    bool IsNumericType() const { return IsFloatType() || IsIntType(); }

    /** Returns the variability of the type. */
    virtual Variability GetVariability() const = 0;

    /** Returns true if the underlying type is uniform */
    bool IsUniformType() const { return GetVariability() == Variability::Uniform; }

    /** Returns true if the underlying type is varying */
    bool IsVaryingType() const { return GetVariability() == Variability::Varying; }

    /** Returns true if the type is laid out in "structure of arrays"
        layout. */
    bool IsSOAType() const { return GetVariability() == Variability::SOA; }

    /** Returns the structure of arrays width for SOA types.  This method
        returns zero for types with non-SOA variability. */
    int GetSOAWidth() const { return GetVariability().soaWidth; }

    /** Returns true if the underlying type's uniform/varying-ness is
        unbound. */
    bool HasUnboundVariability() const { return GetVariability() == Variability::Unbound; }

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

    virtual const Type *GetAsSOAType(int width) const = 0;

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

    /** Returns the LLVM type corresponding to this ispc type. */
    virtual llvm::Type *LLVMType(llvm::LLVMContext *ctx) const = 0;

    /** Returns the LLVM storage type corresponding to this ispc type. */
    virtual llvm::Type *LLVMStorageType(llvm::LLVMContext *ctx) const;

    /** Returns the DIType (LLVM's debugging information structure),
        corresponding to this type. */
    virtual llvm::DIType *GetDIType(llvm::DIScope *scope) const = 0;

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
    static const Type *MoreGeneralType(const Type *type0, const Type *type1, SourcePos pos, const char *reason,
                                       bool forceVarying = false, int vecSize = 0);

    /** Returns true if the given type is an atomic, enum, or pointer type
        (i.e. not an aggregation of multiple instances of a type or
        types.) */
    static bool IsBasicType(const Type *type);

    /** Indicates which Type implementation this type is.  This value can
        be used to determine the actual type much more efficiently than
        using dynamic_cast. */
    const TypeId typeId;

  protected:
    Type(TypeId id) : typeId(id) {}
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
    const AtomicType *GetAsSOAType(int width) const;

    const AtomicType *ResolveUnboundVariability(Variability v) const;
    const AtomicType *GetAsUnsignedType() const;
    const AtomicType *GetAsConstType() const;
    const AtomicType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::Type *LLVMStorageType(llvm::LLVMContext *ctx) const;
    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

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

    static const AtomicType *UniformBool, *VaryingBool;
    static const AtomicType *UniformInt8, *VaryingInt8;
    static const AtomicType *UniformInt16, *VaryingInt16;
    static const AtomicType *UniformInt32, *VaryingInt32;
    static const AtomicType *UniformUInt8, *VaryingUInt8;
    static const AtomicType *UniformUInt16, *VaryingUInt16;
    static const AtomicType *UniformUInt32, *VaryingUInt32;
    static const AtomicType *UniformFloat, *VaryingFloat;
    static const AtomicType *UniformInt64, *VaryingInt64;
    static const AtomicType *UniformUInt64, *VaryingUInt64;
    static const AtomicType *UniformDouble, *VaryingDouble;
    static const AtomicType *Void;

  private:
    const Variability variability;
    const bool isConst;
    AtomicType(BasicType basicType, Variability v, bool isConst);

    mutable const AtomicType *asOtherConstType, *asUniformType, *asVaryingType;
};

/** @brief Type implementation for enumerated types
 *
 *  Note that ISPC enum assumes 32 bit int as underlying type.
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
    const EnumType *GetAsSOAType(int width) const;

    const EnumType *ResolveUnboundVariability(Variability v) const;
    const EnumType *GetAsConstType() const;
    const EnumType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    /** Returns the name of the enum type.  (e.g. struct Foo -> "Foo".) */
    const std::string &GetEnumName() const { return name; }

    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

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

    Pointers can have two additional properties beyond their variability
    and the type of object that they are pointing to.  Both of these
    properties are used for internal bookkeeping and aren't directly
    accessible from the language.

    - Slice: pointers that point to data with SOA layout have this
      property--it indicates that the pointer has two components, where the
      first (major) component is a regular pointer that points to an
      instance of the soa<> type being indexed, and where the second
      (minor) component is an integer that indicates which of the soa
      slices in that instance the pointer points to.

    - Frozen: only slice pointers may have this property--it indicates that
      any further indexing calculations should only be applied to the major
      pointer, and the value of the minor offset should be left unchanged.
      Pointers to lvalues from structure member access have the frozen
      property; see discussion in comments in the StructMemberExpr class.
 */
class PointerType : public Type {
  public:
    PointerType(const Type *t, Variability v, bool isConst, bool isSlice = false, bool frozen = false);

    /** Helper method to return a uniform pointer to the given type. */
    static PointerType *GetUniform(const Type *t, bool isSlice = false);
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

    bool IsSlice() const { return isSlice; }
    bool IsFrozenSlice() const { return isFrozen; }
    const PointerType *GetAsSlice() const;
    const PointerType *GetAsNonSlice() const;
    const PointerType *GetAsFrozenSlice() const;

    const Type *GetBaseType() const;
    const PointerType *GetAsVaryingType() const;
    const PointerType *GetAsUniformType() const;
    const PointerType *GetAsUnboundVariabilityType() const;
    const PointerType *GetAsSOAType(int width) const;

    const PointerType *ResolveUnboundVariability(Variability v) const;
    const PointerType *GetAsConstType() const;
    const PointerType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

    static PointerType *Void;

  private:
    const Variability variability;
    const bool isConst;
    const bool isSlice, isFrozen;
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

  protected:
    CollectionType(TypeId id) : Type(id) {}
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

  protected:
    SequentialType(TypeId id) : CollectionType(id) {}
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
    const ArrayType *GetAsSOAType(int width) const;
    const ArrayType *ResolveUnboundVariability(Variability v) const;

    const ArrayType *GetAsUnsignedType() const;
    const ArrayType *GetAsConstType() const;
    const ArrayType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;
    llvm::ArrayType *LLVMType(llvm::LLVMContext *ctx) const;

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
    /** Type of the elements of the array. */
    const Type *const child;
    /** Number of elements in the array. */
    const int numElements;
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
    const VectorType *GetAsSOAType(int width) const;
    const VectorType *ResolveUnboundVariability(Variability v) const;

    const VectorType *GetAsUnsignedType() const;
    const VectorType *GetAsConstType() const;
    const VectorType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::Type *LLVMStorageType(llvm::LLVMContext *ctx) const;
    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

    int GetElementCount() const;
    const AtomicType *GetElementType() const;

  private:
    /** Base type that the vector holds elements of */
    const AtomicType *const base;
    /** Number of elements in the vector */
    const int numElements;

  public:
    /** Returns the number of elements stored in memory for the vector.
        For uniform vectors, this is rounded up so that the number of
        elements evenly divides the target's native vector width. */
    int getVectorMemoryCount() const;
};

/** @brief Representation of a structure holding a number of members.
 */
class StructType : public CollectionType {
  public:
    StructType(const std::string &name, const llvm::SmallVector<const Type *, 8> &elts,
               const llvm::SmallVector<std::string, 8> &eltNames, const llvm::SmallVector<SourcePos, 8> &eltPositions,
               bool isConst, Variability variability, bool isAnonymous, SourcePos pos);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;
    bool IsDefined() const;

    const Type *GetBaseType() const;
    const StructType *GetAsVaryingType() const;
    const StructType *GetAsUniformType() const;
    const StructType *GetAsUnboundVariabilityType() const;
    const StructType *GetAsSOAType(int width) const;
    const StructType *ResolveUnboundVariability(Variability v) const;

    const StructType *GetAsConstType() const;
    const StructType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

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
    const std::string &GetElementName(int i) const { return elementNames[i]; }

    /** Returns the total number of elements in the structure. */
    int GetElementCount() const { return int(elementTypes.size()); }

    const SourcePos &GetElementPosition(int i) const { return elementPositions[i]; }

    /** Returns the name of the structure type.  (e.g. struct Foo -> "Foo".) */
    const std::string &GetStructName() const { return name; }
    const std::string GetCStructName() const;

  private:
    static bool checkIfCanBeSOA(const StructType *st);

    /*const*/ std::string name;
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
    const llvm::SmallVector<const Type *, 8> elementTypes;
    const llvm::SmallVector<std::string, 8> elementNames;
    /** Source file position at which each structure element declaration
        appeared. */
    const llvm::SmallVector<SourcePos, 8> elementPositions;
    const Variability variability;
    const bool isConst;
    const bool isAnonymous;
    const SourcePos pos;

    mutable llvm::SmallVector<const Type *, 8> finalElementTypes;

    mutable const StructType *oppositeConstStructType;
};

/** Type implementation representing a struct name that has been declared
    but where the struct members haven't been defined (i.e. "struct Foo;").
    This class doesn't do much besides serve as a placeholder that other
    code can use to detect the presence of such as truct.
 */
class UndefinedStructType : public Type {
  public:
    UndefinedStructType(const std::string &name, const Variability variability, bool isConst, SourcePos pos);

    Variability GetVariability() const;

    bool IsBoolType() const;
    bool IsFloatType() const;
    bool IsIntType() const;
    bool IsUnsignedType() const;
    bool IsConstType() const;

    const Type *GetBaseType() const;
    const UndefinedStructType *GetAsVaryingType() const;
    const UndefinedStructType *GetAsUniformType() const;
    const UndefinedStructType *GetAsUnboundVariabilityType() const;
    const UndefinedStructType *GetAsSOAType(int width) const;
    const UndefinedStructType *ResolveUnboundVariability(Variability v) const;

    const UndefinedStructType *GetAsConstType() const;
    const UndefinedStructType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

    /** Returns the name of the structure type.  (e.g. struct Foo -> "Foo".) */
    const std::string &GetStructName() const { return name; }

  private:
    const std::string name;
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
    const Type *GetAsSOAType(int width) const;
    const ReferenceType *ResolveUnboundVariability(Variability v) const;

    const ReferenceType *GetAsConstType() const;
    const ReferenceType *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &name) const;

    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

  private:
    const Type *const targetType;
    mutable const ReferenceType *asOtherConstType;
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
    FunctionType(const Type *returnType, const llvm::SmallVector<const Type *, 8> &argTypes, SourcePos pos);
    FunctionType(const Type *returnType, const llvm::SmallVector<const Type *, 8> &argTypes,
                 const llvm::SmallVector<std::string, 8> &argNames, const llvm::SmallVector<Expr *, 8> &argDefaults,
                 const llvm::SmallVector<SourcePos, 8> &argPos, bool isTask, bool isExported, bool isExternC,
                 bool isUnmasked);

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
    const Type *GetAsSOAType(int width) const;
    const FunctionType *ResolveUnboundVariability(Variability v) const;

    const Type *GetAsConstType() const;
    const Type *GetAsNonConstType() const;

    std::string GetString() const;
    std::string Mangle() const;
    std::string GetCDeclaration(const std::string &fname) const;
    std::string GetCDeclarationForDispatch(const std::string &fname) const;

    llvm::Type *LLVMType(llvm::LLVMContext *ctx) const;

    llvm::DIType *GetDIType(llvm::DIScope *scope) const;

    const Type *GetReturnType() const { return returnType; }

    const std::string GetReturnTypeString() const;

    /** This method returns the LLVM FunctionType that corresponds to this
        function type.  The \c disableMask parameter indicates whether the
        llvm::FunctionType should have the trailing mask parameter, if
        present, removed from the return function signature. */
    llvm::FunctionType *LLVMFunctionType(llvm::LLVMContext *ctx, bool disableMask = false) const;

    int GetNumParameters() const { return (int)paramTypes.size(); }
    const Type *GetParameterType(int i) const;
    Expr *GetParameterDefault(int i) const;
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

    /** Indicates whether the function doesn't take an implicit mask
        parameter (and thus should start execution with an "all on"
        mask). */
    const bool isUnmasked;

    /** Indicates whether this function has been declared to be safe to run
        with an all-off mask. */
    bool isSafe;

    /** If non-negative, this provides a user-supplied override to the cost
        function estimate for the function. */
    int costOverride;

  private:
    const Type *const returnType;

    // The following four vectors should all have the same length (which is
    // in turn the length returned by GetNumParameters()).
    const llvm::SmallVector<const Type *, 8> paramTypes;
    const llvm::SmallVector<std::string, 8> paramNames;
    /** Default values of the function's arguments.  For arguments without
        default values provided, NULL is stored. */
    mutable llvm::SmallVector<Expr *, 8> paramDefaults;
    /** The names provided (if any) with the function arguments in the
        function's signature.  These should only be used for error messages
        and the like and so not affect testing function types for equality,
        etc. */
    const llvm::SmallVector<SourcePos, 8> paramPositions;
};

/* Efficient dynamic casting of Types.  First, we specify a default
   template function that returns NULL, indicating a failed cast, for
   arbitrary types. */
template <typename T> inline const T *CastType(const Type *type) { return NULL; }

/* Now we have template specializaitons for the Types implemented in this
   file.  Each one checks the Type::typeId member and then performs the
   corresponding static cast if it's safe as per the typeId.
 */
template <> inline const AtomicType *CastType(const Type *type) {
    if (type != NULL && type->typeId == ATOMIC_TYPE)
        return (const AtomicType *)type;
    else
        return NULL;
}

template <> inline const EnumType *CastType(const Type *type) {
    if (type != NULL && type->typeId == ENUM_TYPE)
        return (const EnumType *)type;
    else
        return NULL;
}

template <> inline const PointerType *CastType(const Type *type) {
    if (type != NULL && type->typeId == POINTER_TYPE)
        return (const PointerType *)type;
    else
        return NULL;
}

template <> inline const ArrayType *CastType(const Type *type) {
    if (type != NULL && type->typeId == ARRAY_TYPE)
        return (const ArrayType *)type;
    else
        return NULL;
}

template <> inline const VectorType *CastType(const Type *type) {
    if (type != NULL && type->typeId == VECTOR_TYPE)
        return (const VectorType *)type;
    else
        return NULL;
}

template <> inline const SequentialType *CastType(const Type *type) {
    // Note that this function must be updated if other sequential type
    // implementations are added.
    if (type != NULL && (type->typeId == ARRAY_TYPE || type->typeId == VECTOR_TYPE))
        return (const SequentialType *)type;
    else
        return NULL;
}

template <> inline const CollectionType *CastType(const Type *type) {
    // Similarly a new collection type implementation requires updating
    // this function.
    if (type != NULL && (type->typeId == ARRAY_TYPE || type->typeId == VECTOR_TYPE || type->typeId == STRUCT_TYPE))
        return (const CollectionType *)type;
    else
        return NULL;
}

template <> inline const StructType *CastType(const Type *type) {
    if (type != NULL && type->typeId == STRUCT_TYPE)
        return (const StructType *)type;
    else
        return NULL;
}

template <> inline const UndefinedStructType *CastType(const Type *type) {
    if (type != NULL && type->typeId == UNDEFINED_STRUCT_TYPE)
        return (const UndefinedStructType *)type;
    else
        return NULL;
}

template <> inline const ReferenceType *CastType(const Type *type) {
    if (type != NULL && type->typeId == REFERENCE_TYPE)
        return (const ReferenceType *)type;
    else
        return NULL;
}

template <> inline const FunctionType *CastType(const Type *type) {
    if (type != NULL && type->typeId == FUNCTION_TYPE)
        return (const FunctionType *)type;
    else
        return NULL;
}

inline bool IsReferenceType(const Type *t) { return CastType<ReferenceType>(t) != NULL; }

} // namespace ispc
