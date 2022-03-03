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

/** @file type.cpp
    @brief Definitions for classes related to type representation
*/

#include "type.h"
#include "expr.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"

#include <map>
#include <stdio.h>

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/MathExtras.h>

using namespace ispc;

/** Utility routine used in code that prints out declarations; returns true
    if the given name should be printed, false otherwise.  This allows us
    to omit the names for various internal things (whose names start with
    double underscores) and emit anonymous declarations for them instead.
 */

static bool lShouldPrintName(const std::string &name) {
    if (name.size() == 0)
        return false;
    else if (name[0] != '_' && name[0] != '$')
        return true;
    else
        return (name.size() == 1) || (name[1] != '_');
}

/** Utility routine to create a llvm array type of the given number of
    the given element type. */
static llvm::DIType *lCreateDIArray(llvm::DIType *eltType, int count) {

    llvm::Metadata *sub = m->diBuilder->getOrCreateSubrange(0, count);
    std::vector<llvm::Metadata *> subs;
    subs.push_back(sub);
    llvm::DINodeArray subArray = m->diBuilder->getOrCreateArray(subs);
    uint64_t size = eltType->getSizeInBits() * count;
    uint64_t align = eltType->getAlignInBits();

    return m->diBuilder->createArrayType(size, align, eltType, subArray);
}

///////////////////////////////////////////////////////////////////////////
// Variability

std::string Variability::GetString() const {
    switch (type) {
    case Uniform:
        return "uniform";
    case Varying:
        return "varying";
    case SOA: {
        char buf[32];
        snprintf(buf, sizeof(buf), "soa<%d>", soaWidth);
        return std::string(buf);
    }
    case Unbound:
        return "/*unbound*/";
    default:
        FATAL("Unhandled variability");
        return "";
    }
}

std::string Variability::MangleString() const {
    switch (type) {
    case Uniform:
        return "un";
    case Varying:
        return "vy";
    case SOA: {
        char buf[32];
        snprintf(buf, sizeof(buf), "soa<%d>", soaWidth);
        return std::string(buf);
    }
    case Unbound:
        FATAL("Unbound unexpected in Variability::MangleString()");
    default:
        FATAL("Unhandled variability");
        return "";
    }
}

///////////////////////////////////////////////////////////////////////////
// Type
llvm::Type *Type::LLVMStorageType(llvm::LLVMContext *ctx) const { return LLVMType(ctx); }
///////////////////////////////////////////////////////////////////////////
// AtomicType

const AtomicType *AtomicType::UniformBool = new AtomicType(AtomicType::TYPE_BOOL, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingBool = new AtomicType(AtomicType::TYPE_BOOL, Variability::Varying, false);
const AtomicType *AtomicType::UniformInt8 = new AtomicType(AtomicType::TYPE_INT8, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingInt8 = new AtomicType(AtomicType::TYPE_INT8, Variability::Varying, false);
const AtomicType *AtomicType::UniformUInt8 = new AtomicType(AtomicType::TYPE_UINT8, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingUInt8 = new AtomicType(AtomicType::TYPE_UINT8, Variability::Varying, false);
const AtomicType *AtomicType::UniformInt16 = new AtomicType(AtomicType::TYPE_INT16, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingInt16 = new AtomicType(AtomicType::TYPE_INT16, Variability::Varying, false);
const AtomicType *AtomicType::UniformUInt16 = new AtomicType(AtomicType::TYPE_UINT16, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingUInt16 = new AtomicType(AtomicType::TYPE_UINT16, Variability::Varying, false);
const AtomicType *AtomicType::UniformInt32 = new AtomicType(AtomicType::TYPE_INT32, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingInt32 = new AtomicType(AtomicType::TYPE_INT32, Variability::Varying, false);
const AtomicType *AtomicType::UniformUInt32 = new AtomicType(AtomicType::TYPE_UINT32, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingUInt32 = new AtomicType(AtomicType::TYPE_UINT32, Variability::Varying, false);
const AtomicType *AtomicType::UniformFloat16 = new AtomicType(AtomicType::TYPE_FLOAT16, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingFloat16 = new AtomicType(AtomicType::TYPE_FLOAT16, Variability::Varying, false);
const AtomicType *AtomicType::UniformFloat = new AtomicType(AtomicType::TYPE_FLOAT, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingFloat = new AtomicType(AtomicType::TYPE_FLOAT, Variability::Varying, false);
const AtomicType *AtomicType::UniformInt64 = new AtomicType(AtomicType::TYPE_INT64, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingInt64 = new AtomicType(AtomicType::TYPE_INT64, Variability::Varying, false);
const AtomicType *AtomicType::UniformUInt64 = new AtomicType(AtomicType::TYPE_UINT64, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingUInt64 = new AtomicType(AtomicType::TYPE_UINT64, Variability::Varying, false);
const AtomicType *AtomicType::UniformDouble = new AtomicType(AtomicType::TYPE_DOUBLE, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingDouble = new AtomicType(AtomicType::TYPE_DOUBLE, Variability::Varying, false);
const AtomicType *AtomicType::Void = new AtomicType(TYPE_VOID, Variability::Uniform, false);

AtomicType::AtomicType(BasicType bt, Variability v, bool ic)
    : Type(ATOMIC_TYPE), basicType(bt), variability(v), isConst(ic) {
    asOtherConstType = NULL;
    asUniformType = asVaryingType = NULL;
}

Variability AtomicType::GetVariability() const { return variability; }

bool Type::IsPointerType() const { return (CastType<PointerType>(this) != NULL); }

bool Type::IsArrayType() const { return (CastType<ArrayType>(this) != NULL); }

bool Type::IsReferenceType() const { return (CastType<ReferenceType>(this) != NULL); }

bool Type::IsVoidType() const { return EqualIgnoringConst(this, AtomicType::Void); }

bool AtomicType::IsFloatType() const {
    return (basicType == TYPE_FLOAT16 || basicType == TYPE_FLOAT || basicType == TYPE_DOUBLE);
}

bool AtomicType::IsIntType() const {
    return (basicType == TYPE_INT8 || basicType == TYPE_UINT8 || basicType == TYPE_INT16 || basicType == TYPE_UINT16 ||
            basicType == TYPE_INT32 || basicType == TYPE_UINT32 || basicType == TYPE_INT64 || basicType == TYPE_UINT64);
}

bool AtomicType::IsUnsignedType() const {
    return (basicType == TYPE_BOOL || basicType == TYPE_UINT8 || basicType == TYPE_UINT16 || basicType == TYPE_UINT32 ||
            basicType == TYPE_UINT64);
}

bool AtomicType::IsBoolType() const { return basicType == TYPE_BOOL; }

bool AtomicType::IsConstType() const { return isConst; }

const AtomicType *AtomicType::GetAsUnsignedType() const {
    if (IsUnsignedType() == true)
        return this;

    if (IsIntType() == false)
        return NULL;

    switch (basicType) {
    case TYPE_INT8:
        return new AtomicType(TYPE_UINT8, variability, isConst);
    case TYPE_INT16:
        return new AtomicType(TYPE_UINT16, variability, isConst);
    case TYPE_INT32:
        return new AtomicType(TYPE_UINT32, variability, isConst);
    case TYPE_INT64:
        return new AtomicType(TYPE_UINT64, variability, isConst);
    default:
        FATAL("Unexpected basicType in GetAsUnsignedType()");
        return NULL;
    }
}

const AtomicType *AtomicType::GetAsConstType() const {
    if (isConst == true)
        return this;

    if (asOtherConstType == NULL) {
        asOtherConstType = new AtomicType(basicType, variability, true);
        asOtherConstType->asOtherConstType = this;
    }
    return asOtherConstType;
}

const AtomicType *AtomicType::GetAsNonConstType() const {
    if (isConst == false)
        return this;

    if (asOtherConstType == NULL) {
        asOtherConstType = new AtomicType(basicType, variability, false);
        asOtherConstType->asOtherConstType = this;
    }
    return asOtherConstType;
}

const AtomicType *AtomicType::GetBaseType() const { return this; }

const AtomicType *AtomicType::GetAsVaryingType() const {
    Assert(basicType != TYPE_VOID);
    if (variability == Variability::Varying)
        return this;

    if (asVaryingType == NULL) {
        asVaryingType = new AtomicType(basicType, Variability::Varying, isConst);
        if (variability == Variability::Uniform)
            asVaryingType->asUniformType = this;
    }
    return asVaryingType;
}

const AtomicType *AtomicType::GetAsUniformType() const {
    Assert(basicType != TYPE_VOID);
    if (variability == Variability::Uniform)
        return this;

    if (asUniformType == NULL) {
        asUniformType = new AtomicType(basicType, Variability::Uniform, isConst);
        if (variability == Variability::Varying)
            asUniformType->asVaryingType = this;
    }
    return asUniformType;
}

const AtomicType *AtomicType::GetAsUnboundVariabilityType() const {
    Assert(basicType != TYPE_VOID);
    if (variability == Variability::Unbound)
        return this;
    return new AtomicType(basicType, Variability::Unbound, isConst);
}

const AtomicType *AtomicType::GetAsSOAType(int width) const {
    Assert(basicType != TYPE_VOID);
    if (variability == Variability(Variability::SOA, width))
        return this;
    return new AtomicType(basicType, Variability(Variability::SOA, width), isConst);
}

const AtomicType *AtomicType::ResolveUnboundVariability(Variability v) const {
    Assert(v != Variability::Unbound);
    if (variability != Variability::Unbound)
        return this;
    return new AtomicType(basicType, v, isConst);
}

std::string AtomicType::GetString() const {
    std::string ret;
    if (isConst)
        ret += "const ";
    if (basicType != TYPE_VOID) {
        ret += variability.GetString();
        ret += " ";
    }

    switch (basicType) {
    case TYPE_VOID:
        ret += "void";
        break;
    case TYPE_BOOL:
        ret += "bool";
        break;
    case TYPE_INT8:
        ret += "int8";
        break;
    case TYPE_UINT8:
        ret += "unsigned int8";
        break;
    case TYPE_INT16:
        ret += "int16";
        break;
    case TYPE_UINT16:
        ret += "unsigned int16";
        break;
    case TYPE_INT32:
        ret += "int32";
        break;
    case TYPE_UINT32:
        ret += "unsigned int32";
        break;
    case TYPE_FLOAT16:
        ret += "float16";
        break;
    case TYPE_FLOAT:
        ret += "float";
        break;
    case TYPE_INT64:
        ret += "int64";
        break;
    case TYPE_UINT64:
        ret += "unsigned int64";
        break;
    case TYPE_DOUBLE:
        ret += "double";
        break;
    default:
        FATAL("Logic error in AtomicType::GetString()");
    }
    return ret;
}

std::string AtomicType::Mangle() const {
    std::string ret;
    if (isConst)
        ret += "C";
    ret += variability.MangleString();

    switch (basicType) {
    case TYPE_VOID:
        ret += "v";
        break;
    case TYPE_BOOL:
        ret += "b";
        break;
    case TYPE_INT8:
        ret += "t";
        break;
    case TYPE_UINT8:
        ret += "T";
        break;
    case TYPE_INT16:
        ret += "s";
        break;
    case TYPE_UINT16:
        ret += "S";
        break;
    case TYPE_INT32:
        ret += "i";
        break;
    case TYPE_UINT32:
        ret += "u";
        break;
    case TYPE_FLOAT16:
        ret += "h";
        break;
    case TYPE_FLOAT:
        ret += "f";
        break;
    case TYPE_INT64:
        ret += "I";
        break;
    case TYPE_UINT64:
        ret += "U";
        break;
    case TYPE_DOUBLE:
        ret += "d";
        break;
    default:
        FATAL("Logic error in AtomicType::Mangle()");
    }
    return ret;
}

std::string AtomicType::GetCDeclaration(const std::string &name) const {
    std::string ret;
    if (variability == Variability::Unbound) {
        Assert(m->errorCount > 0);
        return ret;
    }
    if (isConst)
        ret += "const ";

    switch (basicType) {
    case TYPE_VOID:
        ret += "void";
        break;
    case TYPE_BOOL:
        ret += "bool";
        break;
    case TYPE_INT8:
        ret += "int8_t";
        break;
    case TYPE_UINT8:
        ret += "uint8_t";
        break;
    case TYPE_INT16:
        ret += "int16_t";
        break;
    case TYPE_UINT16:
        ret += "uint16_t";
        break;
    case TYPE_INT32:
        ret += "int32_t";
        break;
    case TYPE_UINT32:
        ret += "uint32_t";
        break;
    case TYPE_FLOAT16:
        ret += "__fp16";
        break;
    case TYPE_FLOAT:
        ret += "float";
        break;
    case TYPE_INT64:
        ret += "int64_t";
        break;
    case TYPE_UINT64:
        ret += "uint64_t";
        break;
    case TYPE_DOUBLE:
        ret += "double";
        break;
    default:
        FATAL("Logic error in AtomicType::GetCDeclaration()");
    }

    if (lShouldPrintName(name)) {
        ret += " ";
        ret += name;
    }

    if (variability == Variability::SOA) {
        char buf[32];
        snprintf(buf, sizeof(buf), "[%d]", variability.soaWidth);
        ret += buf;
    }

    return ret;
}

static llvm::Type *lGetAtomicLLVMType(llvm::LLVMContext *ctx, const AtomicType *aType, bool isStorageType) {
    Variability variability = aType->GetVariability();
    AtomicType::BasicType basicType = aType->basicType;
    Assert(variability.type != Variability::Unbound);
    bool isUniform = (variability == Variability::Uniform);
    bool isVarying = (variability == Variability::Varying);

    if (isUniform || isVarying) {
        switch (basicType) {
        case AtomicType::TYPE_VOID:
            return llvm::Type::getVoidTy(*ctx);
        case AtomicType::TYPE_BOOL:
            if (isStorageType)
                return isUniform ? LLVMTypes::BoolStorageType : LLVMTypes::BoolVectorStorageType;
            else
                return isUniform ? LLVMTypes::BoolType : LLVMTypes::BoolVectorType;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_UINT8:
            return isUniform ? LLVMTypes::Int8Type : LLVMTypes::Int8VectorType;
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16:
            return isUniform ? LLVMTypes::Int16Type : LLVMTypes::Int16VectorType;
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
            return isUniform ? LLVMTypes::Int32Type : LLVMTypes::Int32VectorType;
        case AtomicType::TYPE_FLOAT16:
            return isUniform ? LLVMTypes::Float16Type : LLVMTypes::Float16VectorType;
        case AtomicType::TYPE_FLOAT:
            return isUniform ? LLVMTypes::FloatType : LLVMTypes::FloatVectorType;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            return isUniform ? LLVMTypes::Int64Type : LLVMTypes::Int64VectorType;
        case AtomicType::TYPE_DOUBLE:
            return isUniform ? LLVMTypes::DoubleType : LLVMTypes::DoubleVectorType;
        default:
            FATAL("logic error in lGetAtomicLLVMType");
            return NULL;
        }
    } else {
        ArrayType at(aType->GetAsUniformType(), variability.soaWidth);
        return at.LLVMType(ctx);
    }
}

llvm::Type *AtomicType::LLVMStorageType(llvm::LLVMContext *ctx) const { return lGetAtomicLLVMType(ctx, this, true); }
llvm::Type *AtomicType::LLVMType(llvm::LLVMContext *ctx) const { return lGetAtomicLLVMType(ctx, this, false); }

llvm::DIType *AtomicType::GetDIType(llvm::DIScope *scope) const {
    Assert(variability.type != Variability::Unbound);

    if (variability.type == Variability::Uniform) {
        switch (basicType) {
        case TYPE_VOID:
            return NULL;

        case TYPE_BOOL:
            return m->diBuilder->createBasicType("bool", 32 /* size */, llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_INT8:
            return m->diBuilder->createBasicType("int8", 8 /* size */, llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT8:
            return m->diBuilder->createBasicType("uint8", 8 /* size */, llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_INT16:
            return m->diBuilder->createBasicType("int16", 16 /* size */, llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT16:
            return m->diBuilder->createBasicType("uint16", 16 /* size */, llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_INT32:
            return m->diBuilder->createBasicType("int32", 32 /* size */, llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT32:
            return m->diBuilder->createBasicType("uint32", 32 /* size */, llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_FLOAT16:
            return m->diBuilder->createBasicType("float16", 16 /* size */, llvm::dwarf::DW_ATE_float);
            break;
        case TYPE_FLOAT:
            return m->diBuilder->createBasicType("float", 32 /* size */, llvm::dwarf::DW_ATE_float);
            break;
        case TYPE_DOUBLE:
            return m->diBuilder->createBasicType("double", 64 /* size */, llvm::dwarf::DW_ATE_float);
            break;
        case TYPE_INT64:
            return m->diBuilder->createBasicType("int64", 64 /* size */, llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT64:
            return m->diBuilder->createBasicType("uint64", 64 /* size */, llvm::dwarf::DW_ATE_unsigned);
            break;

        default:
            FATAL("unhandled basic type in AtomicType::GetDIType()");

            return NULL;
        }
    } else if (variability == Variability::Varying) {

        llvm::Metadata *sub = m->diBuilder->getOrCreateSubrange(0, g->target->getVectorWidth());

        llvm::DINodeArray subArray = m->diBuilder->getOrCreateArray(sub);
        llvm::DIType *unifType = GetAsUniformType()->GetDIType(scope);
        uint64_t size = unifType->getSizeInBits() * g->target->getVectorWidth();
        uint64_t align = unifType->getAlignInBits() * g->target->getVectorWidth();
        return m->diBuilder->createVectorType(size, align, unifType, subArray);
    } else {
        Assert(variability == Variability::SOA);
        ArrayType at(GetAsUniformType(), variability.soaWidth);
        return at.GetDIType(scope);
    }
}

///////////////////////////////////////////////////////////////////////////
// EnumType

EnumType::EnumType(SourcePos p) : Type(ENUM_TYPE), pos(p) {
    //    name = "/* (anonymous) */";
    isConst = false;
    variability = Variability(Variability::Unbound);
}

EnumType::EnumType(const char *n, SourcePos p) : Type(ENUM_TYPE), pos(p), name(n) {
    isConst = false;
    variability = Variability(Variability::Unbound);
}

Variability EnumType::GetVariability() const { return variability; }

bool EnumType::IsBoolType() const { return false; }

bool EnumType::IsFloatType() const { return false; }

bool EnumType::IsIntType() const { return true; }

bool EnumType::IsUnsignedType() const { return true; }

bool EnumType::IsConstType() const { return isConst; }

const EnumType *EnumType::GetBaseType() const { return this; }

const EnumType *EnumType::GetAsUniformType() const {
    if (IsUniformType())
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->variability = Variability::Uniform;
        return enumType;
    }
}

const EnumType *EnumType::ResolveUnboundVariability(Variability v) const {
    if (variability != Variability::Unbound)
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->variability = v;
        return enumType;
    }
}

const EnumType *EnumType::GetAsVaryingType() const {
    if (IsVaryingType())
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->variability = Variability(Variability::Varying);
        return enumType;
    }
}

const EnumType *EnumType::GetAsUnboundVariabilityType() const {
    if (HasUnboundVariability())
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->variability = Variability(Variability::Unbound);
        return enumType;
    }
}

const EnumType *EnumType::GetAsSOAType(int width) const {
    if (GetSOAWidth() == width)
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->variability = Variability(Variability::SOA, width);
        return enumType;
    }
}

const EnumType *EnumType::GetAsConstType() const {
    if (isConst)
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->isConst = true;
        return enumType;
    }
}

const EnumType *EnumType::GetAsNonConstType() const {
    if (!isConst)
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->isConst = false;
        return enumType;
    }
}

std::string EnumType::GetString() const {
    std::string ret;
    if (isConst)
        ret += "const ";
    ret += variability.GetString();

    ret += " enum ";
    if (name.size())
        ret += name;
    return ret;
}

std::string EnumType::Mangle() const {
    Assert(variability != Variability::Unbound);

    std::string ret;
    if (isConst)
        ret += "C";
    ret += variability.MangleString();
    //    ret += std::string("enum[") + name + std::string("]");
    ret += std::string("enum_5B_") + name + std::string("_5D_");
    return ret;
}

std::string EnumType::GetCDeclaration(const std::string &varName) const {
    if (variability == Variability::Unbound) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret;
    if (isConst)
        ret += "const ";
    ret += "enum";
    if (name.size())
        ret += std::string(" ") + name;

    if (lShouldPrintName(varName)) {
        ret += " ";
        ret += varName;
    }

    if (variability == Variability::SOA || variability == Variability::Varying) {
        int vWidth = (variability == Variability::Varying) ? g->target->getVectorWidth() : variability.soaWidth;
        char buf[32];
        snprintf(buf, sizeof(buf), "[%d]", vWidth);
        ret += buf;
    }

    return ret;
}

llvm::Type *EnumType::LLVMType(llvm::LLVMContext *ctx) const {
    Assert(variability != Variability::Unbound);

    switch (variability.type) {
    case Variability::Uniform:
        return LLVMTypes::Int32Type;
    case Variability::Varying:
        return LLVMTypes::Int32VectorType;
    case Variability::SOA: {
        ArrayType at(AtomicType::UniformInt32, variability.soaWidth);
        return at.LLVMType(ctx);
    }
    default:
        FATAL("Unexpected variability in EnumType::LLVMType()");
        return NULL;
    }
}

llvm::DIType *EnumType::GetDIType(llvm::DIScope *scope) const {

    std::vector<llvm::Metadata *> enumeratorDescriptors;
    for (unsigned int i = 0; i < enumerators.size(); ++i) {
        unsigned int enumeratorValue;
        Assert(enumerators[i]->constValue != NULL);
        int count = enumerators[i]->constValue->GetValues(&enumeratorValue);
        Assert(count == 1);

        llvm::Metadata *descriptor = m->diBuilder->createEnumerator(enumerators[i]->name, enumeratorValue);
        enumeratorDescriptors.push_back(descriptor);
    }

    llvm::DINodeArray elementArray = m->diBuilder->getOrCreateArray(enumeratorDescriptors);
    llvm::DIFile *diFile = pos.GetDIFile();
    llvm::DINamespace *diSpace = pos.GetDINamespace();
    llvm::DIType *underlyingType = AtomicType::UniformInt32->GetDIType(scope);
    llvm::DIType *diType =
        m->diBuilder->createEnumerationType(diSpace, GetString(), diFile, pos.first_line, 32 /* size in bits */,
                                            32 /* align in bits */, elementArray, underlyingType, name);
    switch (variability.type) {
    case Variability::Uniform:
        return diType;
    case Variability::Varying: {
        llvm::Metadata *sub = m->diBuilder->getOrCreateSubrange(0, g->target->getVectorWidth());

        llvm::DINodeArray subArray = m->diBuilder->getOrCreateArray(sub);
        // llvm::DebugNodeArray subArray = m->diBuilder->getOrCreateArray(sub);
        uint64_t size = diType->getSizeInBits() * g->target->getVectorWidth();
        uint64_t align = diType->getAlignInBits() * g->target->getVectorWidth();
        return m->diBuilder->createVectorType(size, align, diType, subArray);
    }
    case Variability::SOA: {
        return lCreateDIArray(diType, variability.soaWidth);
    }
    default:
        FATAL("Unexpected variability in EnumType::GetDIType()");
        return NULL;
    }
}

void EnumType::SetEnumerators(const std::vector<Symbol *> &e) { enumerators = e; }

int EnumType::GetEnumeratorCount() const { return (int)enumerators.size(); }

const Symbol *EnumType::GetEnumerator(int i) const { return enumerators[i]; }

///////////////////////////////////////////////////////////////////////////
// PointerType

PointerType *PointerType::Void = new PointerType(AtomicType::Void, Variability(Variability::Uniform), false);

PointerType::PointerType(const Type *t, Variability v, bool ic, bool is, bool fr, AddressSpace as)
    : Type(POINTER_TYPE), variability(v), isConst(ic), isSlice(is), isFrozen(fr), addrSpace(as) {
    baseType = t;
}

PointerType *PointerType::GetUniform(const Type *t, bool is) {
    return new PointerType(t, Variability(Variability::Uniform), false, is);
}

PointerType *PointerType::GetVarying(const Type *t) {
    return new PointerType(t, Variability(Variability::Varying), false);
}

bool PointerType::IsVoidPointer(const Type *t) {
    return Type::EqualIgnoringConst(t->GetAsUniformType(), PointerType::Void);
}

Variability PointerType::GetVariability() const { return variability; }

bool PointerType::IsBoolType() const { return false; }

bool PointerType::IsFloatType() const { return false; }

bool PointerType::IsIntType() const { return false; }

bool PointerType::IsUnsignedType() const { return false; }

bool PointerType::IsConstType() const { return isConst; }

const Type *PointerType::GetBaseType() const { return baseType; }

const PointerType *PointerType::GetAsVaryingType() const {
    if (variability == Variability::Varying)
        return this;
    else
        return new PointerType(baseType, Variability(Variability::Varying), isConst, isSlice, isFrozen);
}

const PointerType *PointerType::GetAsUniformType() const {
    if (variability == Variability::Uniform)
        return this;
    else
        return new PointerType(baseType, Variability(Variability::Uniform), isConst, isSlice, isFrozen);
}

const PointerType *PointerType::GetAsUnboundVariabilityType() const {
    if (variability == Variability::Unbound)
        return this;
    else
        return new PointerType(baseType, Variability(Variability::Unbound), isConst, isSlice, isFrozen);
}

const PointerType *PointerType::GetAsSOAType(int width) const {
    if (GetSOAWidth() == width)
        return this;
    else
        return new PointerType(baseType, Variability(Variability::SOA, width), isConst, isSlice, isFrozen);
}

const PointerType *PointerType::GetAsSlice() const {
    if (isSlice)
        return this;
    return new PointerType(baseType, variability, isConst, true);
}

const PointerType *PointerType::GetAsNonSlice() const {
    if (isSlice == false)
        return this;
    return new PointerType(baseType, variability, isConst, false);
}

const PointerType *PointerType::GetAsFrozenSlice() const {
    if (isFrozen)
        return this;
    return new PointerType(baseType, variability, isConst, true, true);
}

const PointerType *PointerType::GetWithAddrSpace(AddressSpace as) const {
    if (addrSpace == as)
        return this;
    return new PointerType(baseType, variability, isConst, isSlice, isFrozen, as);
}

const PointerType *PointerType::ResolveUnboundVariability(Variability v) const {
    if (baseType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    Assert(v != Variability::Unbound);
    Variability ptrVariability = (variability == Variability::Unbound) ? v : variability;
    const Type *resolvedBaseType = baseType->ResolveUnboundVariability(Variability::Uniform);
    return new PointerType(resolvedBaseType, ptrVariability, isConst, isSlice, isFrozen);
}

const PointerType *PointerType::GetAsConstType() const {
    if (isConst == true)
        return this;
    else
        return new PointerType(baseType, variability, true, isSlice);
}

const PointerType *PointerType::GetAsNonConstType() const {
    if (isConst == false)
        return this;
    else
        return new PointerType(baseType, variability, false, isSlice);
}

std::string PointerType::GetString() const {
    if (baseType == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret = baseType->GetString();

    ret += std::string(" * ");
    if (isConst)
        ret += "const ";
    if (isSlice)
        ret += "slice ";
    if (isFrozen)
        ret += "/*frozen*/ ";
    ret += variability.GetString();

    return ret;
}

std::string PointerType::Mangle() const {
    Assert(variability != Variability::Unbound);
    if (baseType == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret = variability.MangleString() + std::string("_3C_"); // <
    if (isSlice || isFrozen)
        ret += "-";
    if (isSlice)
        ret += "s";
    if (isFrozen)
        ret += "f";
    if (isSlice || isFrozen)
        ret += "-";
    return ret + baseType->Mangle() + std::string("_3E_"); // >
}

std::string PointerType::GetCDeclaration(const std::string &name) const {
    if (isSlice || (variability == Variability::Unbound)) {
        Assert(m->errorCount > 0);
        return "";
    }

    if (baseType == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }

    bool baseIsBasicVarying = (IsBasicType(baseType)) && (baseType->IsVaryingType());
    bool baseIsFunction = (CastType<FunctionType>(baseType) != NULL);

    std::string tempName;
    if (baseIsBasicVarying || baseIsFunction)
        tempName += std::string("(");
    tempName += std::string(" *");
    if (isConst)
        tempName += " const";
    tempName += std::string(" ");
    tempName += name;
    if (baseIsBasicVarying || baseIsFunction)
        tempName += std::string(")");

    std::string ret;
    if (!baseIsFunction) {
        ret = baseType->GetCDeclaration("");
        ret += tempName;
    } else {
        ret += baseType->GetCDeclaration(tempName);
    }
    if (variability == Variability::SOA) {
        char buf[32];
        snprintf(buf, sizeof(buf), "[%d]", variability.soaWidth);
        ret += buf;
    }
    if (baseIsBasicVarying) {
        int vWidth = g->target->getVectorWidth();
        char buf[32];
        snprintf(buf, sizeof(buf), "[%d]", vWidth);
        ret += buf;
    }

    return ret;
}

llvm::Type *PointerType::LLVMType(llvm::LLVMContext *ctx) const {
    if (baseType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    if (isSlice) {
        llvm::Type *types[2];
        types[0] = GetAsNonSlice()->LLVMStorageType(ctx);

        switch (variability.type) {
        case Variability::Uniform:
            types[1] = LLVMTypes::Int32Type;
            break;
        case Variability::Varying:
            types[1] = LLVMTypes::Int32VectorType;
            break;
        case Variability::SOA:
            types[1] = llvm::ArrayType::get(LLVMTypes::Int32Type, variability.soaWidth);
            break;
        default:
            FATAL("unexpected variability for slice pointer in "
                  "PointerType::LLVMType");
        }

        llvm::ArrayRef<llvm::Type *> typesArrayRef = llvm::ArrayRef<llvm::Type *>(types, 2);
        return llvm::StructType::get(*g->ctx, typesArrayRef);
    }

    switch (variability.type) {
    case Variability::Uniform: {
        llvm::Type *ptype = NULL;
        const FunctionType *ftype = CastType<FunctionType>(baseType);
        if (ftype != NULL)
            ptype = llvm::PointerType::get(ftype->LLVMFunctionType(ctx), (unsigned)addrSpace);
        else {
            if (baseType->IsVoidType())
                ptype = llvm::PointerType::get(llvm::Type::getInt8Ty(*ctx), (unsigned)addrSpace);
            else
                ptype = llvm::PointerType::get(baseType->LLVMStorageType(ctx), (unsigned)addrSpace);
        }
        return ptype;
    }
    case Variability::Varying:
        // always the same, since we currently use int vectors for varying
        // pointers
        return LLVMTypes::VoidPointerVectorType;
    case Variability::SOA: {
        ArrayType at(GetAsUniformType(), variability.soaWidth);
        return at.LLVMType(ctx);
    }
    default:
        FATAL("Unexpected variability in PointerType::LLVMType()");
        return NULL;
    }
}

llvm::DIType *PointerType::GetDIType(llvm::DIScope *scope) const {
    if (baseType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    llvm::DIType *diTargetType = baseType->GetDIType(scope);
    int bitsSize = g->target->is32Bit() ? 32 : 64;
    int ptrAlignBits = bitsSize;
    switch (variability.type) {
    case Variability::Uniform:
        return m->diBuilder->createPointerType(diTargetType, bitsSize, ptrAlignBits);
    case Variability::Varying: {
        // emit them as an array of pointers
        llvm::DIDerivedType *eltType = m->diBuilder->createPointerType(diTargetType, bitsSize, ptrAlignBits);
        return lCreateDIArray(eltType, g->target->getVectorWidth());
    }
    case Variability::SOA: {
        ArrayType at(GetAsUniformType(), variability.soaWidth);
        return at.GetDIType(scope);
    }
    default:
        FATAL("Unexpected variability in PointerType::GetDIType()");
        return NULL;
    }
}

///////////////////////////////////////////////////////////////////////////
// SequentialType

const Type *SequentialType::GetElementType(int index) const { return GetElementType(); }

///////////////////////////////////////////////////////////////////////////
// ArrayType

ArrayType::ArrayType(const Type *c, int a) : SequentialType(ARRAY_TYPE), child(c), numElements(a) {
    // 0 -> unsized array.
    Assert(numElements >= 0);
    Assert(c->IsVoidType() == false);
}

llvm::ArrayType *ArrayType::LLVMType(llvm::LLVMContext *ctx) const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Type *ct = child->LLVMStorageType(ctx);
    if (ct == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return llvm::ArrayType::get(ct, numElements);
}

Variability ArrayType::GetVariability() const {
    return child ? child->GetVariability() : Variability(Variability::Uniform);
}

bool ArrayType::IsFloatType() const { return false; }

bool ArrayType::IsIntType() const { return false; }

bool ArrayType::IsUnsignedType() const { return false; }

bool ArrayType::IsBoolType() const { return false; }

bool ArrayType::IsConstType() const { return child ? child->IsConstType() : false; }

const Type *ArrayType::GetBaseType() const {
    const Type *type = child;
    const ArrayType *at = CastType<ArrayType>(type);
    // Keep walking until we reach a child that isn't itself an array
    while (at) {
        type = at->child;
        at = CastType<ArrayType>(type);
    }
    return type;
}

const ArrayType *ArrayType::GetAsVaryingType() const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->GetAsVaryingType(), numElements);
}

const ArrayType *ArrayType::GetAsUniformType() const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->GetAsUniformType(), numElements);
}

const ArrayType *ArrayType::GetAsUnboundVariabilityType() const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->GetAsUnboundVariabilityType(), numElements);
}

const ArrayType *ArrayType::GetAsSOAType(int width) const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->GetAsSOAType(width), numElements);
}

const ArrayType *ArrayType::ResolveUnboundVariability(Variability v) const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->ResolveUnboundVariability(v), numElements);
}

const ArrayType *ArrayType::GetAsUnsignedType() const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->GetAsUnsignedType(), numElements);
}

const ArrayType *ArrayType::GetAsConstType() const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->GetAsConstType(), numElements);
}

const ArrayType *ArrayType::GetAsNonConstType() const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ArrayType(child->GetAsNonConstType(), numElements);
}

int ArrayType::GetElementCount() const { return numElements; }

const Type *ArrayType::GetElementType() const { return child; }

std::string ArrayType::GetString() const {
    const Type *base = GetBaseType();
    if (base == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }
    std::string s = base->GetString();

    const ArrayType *at = this;
    Assert(at);
    // Walk through this and any children arrays and print all of their
    // dimensions
    while (at) {
        char buf[16];
        if (at->numElements > 0)
            snprintf(buf, sizeof(buf), "%d", at->numElements);
        else
            buf[0] = '\0';
        s += std::string("[") + std::string(buf) + std::string("]");
        at = CastType<ArrayType>(at->child);
    }
    return s;
}

std::string ArrayType::Mangle() const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return "(error)";
    }
    std::string s = child->Mangle();
    char buf[16];
    if (numElements > 0)
        snprintf(buf, sizeof(buf), "%d", numElements);
    else
        buf[0] = '\0';
    //    return s + "[" + buf + "]";
    return s + "_5B_" + buf + "_5D_";
}

std::string ArrayType::GetCDeclaration(const std::string &name) const {
    const Type *base = GetBaseType();
    if (base == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }

    int soaWidth = base->GetSOAWidth();
    int vWidth = (base->IsVaryingType()) ? g->target->getVectorWidth() : 0;
    base = base->GetAsUniformType();

    std::string s = base->GetCDeclaration(name);

    const ArrayType *at = this;
    Assert(at);
    while (at) {
        char buf[16];
        if (at->numElements > 0)
            snprintf(buf, sizeof(buf), "%d", at->numElements);
        else
            buf[0] = '\0';
        s += std::string("[") + std::string(buf) + std::string("]");
        at = CastType<ArrayType>(at->child);
    }

    if (soaWidth > 0) {
        char buf[16];
        snprintf(buf, sizeof(buf), "[%d]", soaWidth);
        s += buf;
    }

    if (vWidth > 0) {
        char buf[16];
        snprintf(buf, sizeof(buf), "[%d]", vWidth);
        s += buf;
    }

    return s;
}

int ArrayType::TotalElementCount() const {
    const ArrayType *ct = CastType<ArrayType>(child);
    if (ct != NULL)
        return numElements * ct->TotalElementCount();
    else
        return numElements;
}

llvm::DIType *ArrayType::GetDIType(llvm::DIScope *scope) const {
    if (child == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    llvm::DIType *eltType = child->GetDIType(scope);
    return lCreateDIArray(eltType, numElements);
}

ArrayType *ArrayType::GetSizedArray(int sz) const {
    Assert(numElements == 0);
    return new ArrayType(child, sz);
}

const Type *ArrayType::SizeUnsizedArrays(const Type *type, Expr *initExpr) {
    const ArrayType *at = CastType<ArrayType>(type);
    if (at == NULL)
        return type;

    ExprList *exprList = llvm::dyn_cast_or_null<ExprList>(initExpr);
    if (exprList == NULL || exprList->exprs.size() == 0)
        return type;

    // If the current dimension is unsized, then size it according to the
    // length of the expression list
    if (at->GetElementCount() == 0) {
        type = at->GetSizedArray(exprList->exprs.size());
        at = CastType<ArrayType>(type);
    }

    // Is there another nested level of expression lists?  If not, bail out
    // now.  Otherwise we'll use the first one to size the next dimension
    // (after checking below that it has the same length as all of the
    // other ones.
    ExprList *nextList = llvm::dyn_cast_or_null<ExprList>(exprList->exprs[0]);
    if (nextList == NULL)
        return type;

    const Type *nextType = at->GetElementType();
    const ArrayType *nextArrayType = CastType<ArrayType>(nextType);
    if (nextArrayType != NULL && nextArrayType->GetElementCount() == 0) {
        // If the recursive call to SizeUnsizedArrays at the bottom of the
        // function is going to size an unsized dimension, make sure that
        // all of the sub-expression lists are the same length--i.e. issue
        // an error if we have something like
        // int x[][] = { { 1 }, { 1, 2, 3, 4 } };
        unsigned int nextSize = nextList->exprs.size();
        for (unsigned int i = 1; i < exprList->exprs.size(); ++i) {
            if (exprList->exprs[i] == NULL) {
                // We should have seen an error earlier in this case.
                Assert(m->errorCount > 0);
                continue;
            }

            ExprList *el = llvm::dyn_cast_or_null<ExprList>(exprList->exprs[i]);
            if (el == NULL || el->exprs.size() != nextSize) {
                Error(Union(exprList->exprs[0]->pos, exprList->exprs[i]->pos),
                      "Inconsistent initializer expression list lengths "
                      "make it impossible to size unsized array dimensions.");
                return NULL;
            }
        }
    }

    // Recursively call SizeUnsizedArrays() to get the child type for the
    // array that we were able to size here.
    return new ArrayType(SizeUnsizedArrays(at->GetElementType(), nextList), at->GetElementCount());
}

///////////////////////////////////////////////////////////////////////////
// VectorType

VectorType::VectorType(const AtomicType *b, int a) : SequentialType(VECTOR_TYPE), base(b), numElements(a) {
    Assert(numElements > 0);
    Assert(base != NULL);
}

Variability VectorType::GetVariability() const { return base->GetVariability(); }

bool VectorType::IsFloatType() const { return base->IsFloatType(); }

bool VectorType::IsIntType() const { return base->IsIntType(); }

bool VectorType::IsUnsignedType() const { return base->IsUnsignedType(); }

bool VectorType::IsBoolType() const { return base->IsBoolType(); }

bool VectorType::IsConstType() const { return base->IsConstType(); }

const Type *VectorType::GetBaseType() const { return base; }

const VectorType *VectorType::GetAsVaryingType() const { return new VectorType(base->GetAsVaryingType(), numElements); }

const VectorType *VectorType::GetAsUniformType() const { return new VectorType(base->GetAsUniformType(), numElements); }

const VectorType *VectorType::GetAsUnboundVariabilityType() const {
    return new VectorType(base->GetAsUnboundVariabilityType(), numElements);
}

const VectorType *VectorType::GetAsSOAType(int width) const {
    return new VectorType(base->GetAsSOAType(width), numElements);
}

const VectorType *VectorType::ResolveUnboundVariability(Variability v) const {
    return new VectorType(base->ResolveUnboundVariability(v), numElements);
}

const VectorType *VectorType::GetAsUnsignedType() const {
    if (base == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new VectorType(base->GetAsUnsignedType(), numElements);
}

const VectorType *VectorType::GetAsConstType() const { return new VectorType(base->GetAsConstType(), numElements); }

const VectorType *VectorType::GetAsNonConstType() const {
    return new VectorType(base->GetAsNonConstType(), numElements);
}

std::string VectorType::GetString() const {
    std::string s = base->GetString();
    char buf[16];
    snprintf(buf, sizeof(buf), "<%d>", numElements);
    return s + std::string(buf);
}

std::string VectorType::Mangle() const {
    std::string s = base->Mangle();
    char buf[16];
    snprintf(buf, sizeof(buf), "_3C_%d_3E_", numElements); // "<%d>"
    return s + std::string(buf);
}

std::string VectorType::GetCDeclaration(const std::string &name) const {
    std::string s = base->GetCDeclaration("");
    char buf[16];
    snprintf(buf, sizeof(buf), "%d", numElements);
    return s + std::string(buf) + "  " + name;
}

int VectorType::GetElementCount() const { return numElements; }

const AtomicType *VectorType::GetElementType() const { return base; }

static llvm::Type *lGetVectorLLVMType(llvm::LLVMContext *ctx, const VectorType *vType, bool isStorage) {

    const Type *base = vType->GetBaseType();
    int numElements = vType->GetElementCount();

    if (base == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Type *bt;
    // Non-uniform vector types are represented in IR as an array.
    // So, creating them with base as storage type similar to arrays.
    if (isStorage || !base->IsUniformType())
        bt = base->LLVMStorageType(ctx);
    else
        bt = base->LLVMType(ctx);
    if (!bt)
        return NULL;

    if (base->IsUniformType())
        // Vectors of uniform types are laid out across LLVM vectors, with
        // the llvm vector size set to be a power of 2 bits in size but not less then 128 bit.
        // This is a roundabout way of ensuring that LLVM lays
        // them out into machine vector registers for the specified target
        // so that e.g. if we want to add two uniform 4 float
        // vectors, that is turned into a single addps on SSE.
        return LLVMVECTOR::get(bt, vType->getVectorMemoryCount());
    else if (base->IsVaryingType())
        // varying types are already laid out to fill HW vector registers,
        // so a vector type here is just expanded out as an llvm array.
        return llvm::ArrayType::get(bt, vType->getVectorMemoryCount());
    else if (base->IsSOAType())
        return llvm::ArrayType::get(bt, numElements);
    else {
        FATAL("Unexpected variability in lGetVectorLLVMType()");
        return NULL;
    }
}

llvm::Type *VectorType::LLVMStorageType(llvm::LLVMContext *ctx) const { return lGetVectorLLVMType(ctx, this, true); }

llvm::Type *VectorType::LLVMType(llvm::LLVMContext *ctx) const { return lGetVectorLLVMType(ctx, this, false); }

llvm::DIType *VectorType::GetDIType(llvm::DIScope *scope) const {
    llvm::DIType *eltType = base->GetDIType(scope);

    llvm::Metadata *sub = m->diBuilder->getOrCreateSubrange(0, numElements);

    // vectors of varying types are already naturally aligned to the
    // machine's vector width, but arrays of uniform types need to be
    // explicitly aligned to the machines natural vector alignment.

    llvm::DINodeArray subArray = m->diBuilder->getOrCreateArray(sub);
    uint64_t sizeBits = eltType->getSizeInBits() * numElements;
    uint64_t align = eltType->getAlignInBits();

    if (IsUniformType()) {
        llvm::Type *ty = this->LLVMType(g->ctx);
        align = g->target->getDataLayout()->getABITypeAlignment(ty);
    }

    if (IsUniformType() || IsVaryingType())
        return m->diBuilder->createVectorType(sizeBits, align, eltType, subArray);
    else if (IsSOAType()) {
        ArrayType at(base, numElements);
        return at.GetDIType(scope);
    } else {
        FATAL("Unexpected variability in VectorType::GetDIType()");
        return NULL;
    }
}

int VectorType::getVectorMemoryCount() const {
    if (base->IsVaryingType())
        return numElements;
    else if (base->IsUniformType()) {
        // Round up the element count to power of 2 bits in size but not less then 128 bit in total vector size
        // where one element size is data type width in bits.
        // This strategy was chosen by the following reasons:
        // 1. We need to return the same number of vector elements regardless the element size for correct work of the
        // language.
        // 2. Using next power of two, but not less than 128 bit in total vector size ensures that machine vector
        // registers are used. It generally leads to better performance. This strategy also matches OpenCL short
        // vectors.
        // 3. Using data type width of the target to determine element size makes optimization trade off.
        int nextPow2 = llvm::NextPowerOf2(numElements - 1);
        return (nextPow2 * g->target->getDataTypeWidth() < 128) ? (128 / g->target->getDataTypeWidth()) : nextPow2;
    } else if (base->IsSOAType()) {
        FATAL("VectorType SOA getVectorMemoryCount");
        return -1;
    } else {
        FATAL("Unexpected variability in VectorType::getVectorMemoryCount()");
        return -1;
    }
}

///////////////////////////////////////////////////////////////////////////
// StructType

/** Using a struct's name, its variability, and the vector width for the
    current compilation target, this function generates a string that
    encodes that full structure type, for use in the structTypeMap.  Note
    that the vector width is needed in order to differentiate between
    'varying' structs with different compilation targets, which have
    different memory layouts...
 */
static std::string lMangleStructName(const std::string &name, Variability variability) {
    char buf[32];
    std::string n;

    // Encode vector width
    snprintf(buf, sizeof(buf), "v%d", g->target->getVectorWidth());

    n += buf;

    // Variability
    switch (variability.type) {
    case Variability::Uniform:
        n += "_uniform_";
        break;
    case Variability::Varying:
        n += "_varying_";
        break;
    case Variability::SOA:
        snprintf(buf, sizeof(buf), "_soa%d_", variability.soaWidth);
        n += buf;
        break;
    default:
        FATAL("Unexpected variability in lMangleStructName()");
    }

    // And stuff the name at the end....
    n += name;
    return n;
}

StructType::StructType(const std::string &n, const llvm::SmallVector<const Type *, 8> &elts,
                       const llvm::SmallVector<std::string, 8> &en, const llvm::SmallVector<SourcePos, 8> &ep, bool ic,
                       Variability v, bool ia, SourcePos p)
    : CollectionType(STRUCT_TYPE), name(n), elementTypes(elts), elementNames(en), elementPositions(ep), variability(v),
      isConst(ic), isAnonymous(ia), pos(p) {
    oppositeConstStructType = NULL;
    finalElementTypes.resize(elts.size(), NULL);

    static int count = 0;
    if (variability != Variability::Unbound) {
        // For structs with non-unbound variability, we'll create the
        // correspoing LLVM struct type now, if one hasn't been made
        // already.

        // Create a unique anonymous struct name if we have an anonymous
        // struct (name == "").
        if (name == "") {
            char buf[16];
            snprintf(buf, sizeof(buf), "$anon%d", count);
            name = std::string(buf);
            ++count;
        }

        // If a non-opaque LLVM struct for this type has already been
        // created, we're done.  For an opaque struct type, we'll override
        // the old definition now that we have a full definition.
        std::string mname = lMangleStructName(name, variability);
        if (m->structTypeMap.find(mname) != m->structTypeMap.end() && m->structTypeMap[mname]->isOpaque() == false)
            return;

        // Actually make the LLVM struct
        std::vector<llvm::Type *> elementTypes;
        int nElements = GetElementCount();
        if (nElements == 0) {
            elementTypes.push_back(LLVMTypes::Int8Type);
        } else {
            for (int i = 0; i < nElements; ++i) {
                const Type *type = GetElementType(i);
                if (type == NULL) {
                    Assert(m->errorCount > 0);
                    return;
                } else if (CastType<FunctionType>(type) != NULL) {
                    Error(elementPositions[i], "Method declarations are not "
                                               "supported.");
                    return;
                } else
                    elementTypes.push_back(type->LLVMStorageType(g->ctx));
            }
        }

        if (m->structTypeMap.find(mname) == m->structTypeMap.end()) {
            // New struct definition
            llvm::StructType *st = llvm::StructType::create(*g->ctx, elementTypes, mname);
            m->structTypeMap[mname] = st;
        } else {
            // Definition for what was before just a declaration
            m->structTypeMap[mname]->setBody(elementTypes);
        }
    }
    // Create a unique anonymous struct name if we have an anonymous struct (name == "")
    // Ensuring struct is created with a name, prevents each use of original
    // struct from having different names causing type match errors.
    if (name == "") {
        char buf[16];
        snprintf(buf, sizeof(buf), "$anon%d", count);
        name = std::string(buf);
        ++count;
    }
}

const std::string StructType::GetCStructName() const {
    // only return mangled name for varying structs for backwards
    // compatibility...

    if (variability == Variability::Varying) {
        return lMangleStructName(name, variability);
    } else {
        return GetStructName();
    }
}

Variability StructType::GetVariability() const { return variability; }

bool StructType::IsBoolType() const { return false; }

bool StructType::IsFloatType() const { return false; }

bool StructType::IsIntType() const { return false; }

bool StructType::IsUnsignedType() const { return false; }

bool StructType::IsConstType() const { return isConst; }

bool StructType::IsDefined() const {
    for (int i = 0; i < GetElementCount(); i++) {
        const Type *t = GetElementType(i);
        const UndefinedStructType *ust = CastType<UndefinedStructType>(t);
        if (ust != NULL) {
            return false;
        }
        const StructType *st = CastType<StructType>(t);
        if (st != NULL) {
            if (!st->IsDefined()) {
                return false;
            }
        }
    }
    return true;
}

const Type *StructType::GetBaseType() const { return this; }

const StructType *StructType::GetAsVaryingType() const {
    if (IsVaryingType())
        return this;
    else
        return new StructType(name, elementTypes, elementNames, elementPositions, isConst,
                              Variability(Variability::Varying), isAnonymous, pos);
}

const StructType *StructType::GetAsUniformType() const {
    if (IsUniformType())
        return this;
    else
        return new StructType(name, elementTypes, elementNames, elementPositions, isConst,
                              Variability(Variability::Uniform), isAnonymous, pos);
}

const StructType *StructType::GetAsUnboundVariabilityType() const {
    if (HasUnboundVariability())
        return this;
    else
        return new StructType(name, elementTypes, elementNames, elementPositions, isConst,
                              Variability(Variability::Unbound), isAnonymous, pos);
}

const StructType *StructType::GetAsSOAType(int width) const {
    if (GetSOAWidth() == width)
        return this;

    if (checkIfCanBeSOA(this) == false)
        return NULL;

    return new StructType(name, elementTypes, elementNames, elementPositions, isConst,
                          Variability(Variability::SOA, width), isAnonymous, pos);
}

const StructType *StructType::ResolveUnboundVariability(Variability v) const {
    Assert(v != Variability::Unbound);

    if (variability != Variability::Unbound)
        return this;

    // We don't resolve the members here but leave them unbound, so that if
    // resolve to varying but later want to get the uniform version of this
    // type, for example, then we still have the information around about
    // which element types were originally unbound...
    return new StructType(name, elementTypes, elementNames, elementPositions, isConst, v, isAnonymous, pos);
}

const StructType *StructType::GetAsConstType() const {
    if (isConst == true)
        return this;
    else if (oppositeConstStructType != NULL)
        return oppositeConstStructType;
    else {
        oppositeConstStructType =
            new StructType(name, elementTypes, elementNames, elementPositions, true, variability, isAnonymous, pos);
        oppositeConstStructType->oppositeConstStructType = this;
        return oppositeConstStructType;
    }
}

const StructType *StructType::GetAsNonConstType() const {
    if (isConst == false)
        return this;
    else if (oppositeConstStructType != NULL)
        return oppositeConstStructType;
    else {
        oppositeConstStructType =
            new StructType(name, elementTypes, elementNames, elementPositions, false, variability, isAnonymous, pos);
        oppositeConstStructType->oppositeConstStructType = this;
        return oppositeConstStructType;
    }
}

std::string StructType::GetString() const {
    std::string ret;
    if (isConst)
        ret += "const ";
    ret += variability.GetString();
    ret += " struct ";

    if (isAnonymous) {
        // Print the whole anonymous struct declaration
        ret += name + std::string(" { ");
        for (unsigned int i = 0; i < elementTypes.size(); ++i) {
            ret += elementTypes[i]->GetString();
            ret += " ";
            ret += elementNames[i];
            ret += "; ";
        }
        ret += "}";
    } else {
        ret += name;
    }

    return ret;
}

/** Mangle a struct name for use in function name mangling. */
static std::string lMangleStruct(Variability variability, bool isConst, const std::string &name) {
    Assert(variability != Variability::Unbound);

    std::string ret;
    //    ret += "s[";
    ret += "s_5B_";
    if (isConst)
        ret += "_c_";
    ret += variability.MangleString();

    //    ret += name + std::string("]");
    ret += name + std::string("_5D_");
    return ret;
}

std::string StructType::Mangle() const { return lMangleStruct(variability, isConst, name); }

std::string StructType::GetCDeclaration(const std::string &n) const {
    std::string ret;
    if (isConst)
        ret += "const ";
    ret += std::string("struct ") + GetCStructName();

    // Add _SOA<SOAWIDTH> to end of struct name.
    if (variability.soaWidth > 0) {
        char buf[32];
        // This has to match the naming scheme used in lEmitStructDecls()
        // in module.cpp
        snprintf(buf, sizeof(buf), "_SOA%d", variability.soaWidth);
        ret += buf;
    }

    if (lShouldPrintName(n)) {
        ret += std::string(" ") + n;
    }

    return ret;
}

llvm::Type *StructType::LLVMType(llvm::LLVMContext *ctx) const {
    Assert(variability != Variability::Unbound);
    std::string mname = lMangleStructName(name, variability);
    if (m->structTypeMap.find(mname) == m->structTypeMap.end()) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return m->structTypeMap[mname];
}

// Versioning of this function becomes really messy, so versioning the whole function.
llvm::DIType *StructType::GetDIType(llvm::DIScope *scope) const {
    llvm::Type *llvm_type = LLVMStorageType(g->ctx);
    auto &dataLayout = m->module->getDataLayout();
    auto layout = dataLayout.getStructLayout(llvm::dyn_cast_or_null<llvm::StructType>(llvm_type));
    std::vector<llvm::Metadata *> elementLLVMTypes;
    // Walk through the elements of the struct; for each one figure out its
    // alignment and size, using that to figure out its offset w.r.t. the
    // start of the structure.
    for (unsigned int i = 0; i < elementTypes.size(); ++i) {
        llvm::DIType *eltType = GetElementType(i)->GetDIType(scope);
        uint64_t eltSize = eltType->getSizeInBits();

        auto llvmType = GetElementType(i)->LLVMStorageType(g->ctx);
        uint64_t eltAlign = dataLayout.getABITypeAlignment(llvmType) * 8;
        Assert(eltAlign != 0);

        auto eltOffset = layout->getElementOffsetInBits(i);

        int line = elementPositions[i].first_line;
        llvm::DIFile *diFile = elementPositions[i].GetDIFile();
        llvm::DIDerivedType *fieldType = m->diBuilder->createMemberType(
            scope, elementNames[i], diFile, line, eltSize, eltAlign, eltOffset, llvm::DINode::FlagZero, eltType);
        elementLLVMTypes.push_back(fieldType);
    }

    llvm::DINodeArray elements = m->diBuilder->getOrCreateArray(elementLLVMTypes);
    llvm::DIFile *diFile = pos.GetDIFile();
    llvm::DINamespace *diSpace = pos.GetDINamespace();
    return m->diBuilder->createStructType(diSpace, GetString(), diFile,
                                          pos.first_line,                     // Line number
                                          layout->getSizeInBits(),            // Size in bits
                                          layout->getAlignment().value() * 8, // Alignment in bits
                                          llvm::DINode::FlagZero,             // Flags
                                          NULL, elements);
}

const Type *StructType::GetElementType(int i) const {
    Assert(variability != Variability::Unbound);
    Assert(i < (int)elementTypes.size());

    if (finalElementTypes[i] == NULL) {
        const Type *type = elementTypes[i];
        if (type == NULL) {
            Assert(m->errorCount > 0);
            return NULL;
        }

        // If the element has unbound variability, resolve its variability to
        // the struct type's variability
        type = type->ResolveUnboundVariability(variability);
        if (isConst)
            type = type->GetAsConstType();
        finalElementTypes[i] = type;
    }

    return finalElementTypes[i];
}

const Type *StructType::GetElementType(const std::string &n) const {
    for (unsigned int i = 0; i < elementNames.size(); ++i)
        if (elementNames[i] == n)
            return GetElementType(i);
    return NULL;
}

int StructType::GetElementNumber(const std::string &n) const {
    for (unsigned int i = 0; i < elementNames.size(); ++i)
        if (elementNames[i] == n)
            return i;
    return -1;
}

bool StructType::checkIfCanBeSOA(const StructType *st) {
    bool ok = true;
    for (int i = 0; i < (int)st->elementTypes.size(); ++i) {
        const Type *eltType = st->elementTypes[i];
        const StructType *childStructType = CastType<StructType>(eltType);

        if (childStructType != NULL)
            ok &= checkIfCanBeSOA(childStructType);
        else if (eltType->HasUnboundVariability() == false) {
            Error(st->elementPositions[i],
                  "Unable to apply SOA conversion to "
                  "struct due to \"%s\" member \"%s\" with bound \"%s\" "
                  "variability.",
                  eltType->GetString().c_str(), st->elementNames[i].c_str(),
                  eltType->IsUniformType() ? "uniform" : "varying");
            ok = false;
        } else if (CastType<ReferenceType>(eltType)) {
            Error(st->elementPositions[i],
                  "Unable to apply SOA conversion to "
                  "struct due to member \"%s\" with reference type \"%s\".",
                  st->elementNames[i].c_str(), eltType->GetString().c_str());
            ok = false;
        }
    }
    return ok;
}

///////////////////////////////////////////////////////////////////////////
// UndefinedStructType

UndefinedStructType::UndefinedStructType(const std::string &n, const Variability var, bool ic, SourcePos p)
    : Type(UNDEFINED_STRUCT_TYPE), name(n), variability(var), isConst(ic), pos(p) {
    Assert(name != "");
    if (variability != Variability::Unbound) {
        // Create a new opaque LLVM struct type for this struct name
        std::string mname = lMangleStructName(name, variability);
        if (m->structTypeMap.find(mname) == m->structTypeMap.end())
            m->structTypeMap[mname] = llvm::StructType::create(*g->ctx, mname);
    }
}

Variability UndefinedStructType::GetVariability() const { return variability; }

bool UndefinedStructType::IsBoolType() const { return false; }

bool UndefinedStructType::IsFloatType() const { return false; }

bool UndefinedStructType::IsIntType() const { return false; }

bool UndefinedStructType::IsUnsignedType() const { return false; }

bool UndefinedStructType::IsConstType() const { return isConst; }

const Type *UndefinedStructType::GetBaseType() const { return this; }

const UndefinedStructType *UndefinedStructType::GetAsVaryingType() const {
    if (variability == Variability::Varying)
        return this;
    return new UndefinedStructType(name, Variability::Varying, isConst, pos);
}

const UndefinedStructType *UndefinedStructType::GetAsUniformType() const {
    if (variability == Variability::Uniform)
        return this;
    return new UndefinedStructType(name, Variability::Uniform, isConst, pos);
}

const UndefinedStructType *UndefinedStructType::GetAsUnboundVariabilityType() const {
    if (variability == Variability::Unbound)
        return this;
    return new UndefinedStructType(name, Variability::Unbound, isConst, pos);
}

const UndefinedStructType *UndefinedStructType::GetAsSOAType(int width) const {
    FATAL("UndefinedStructType::GetAsSOAType() shouldn't be called.");
    return NULL;
}

const UndefinedStructType *UndefinedStructType::ResolveUnboundVariability(Variability v) const {
    if (variability != Variability::Unbound)
        return this;
    return new UndefinedStructType(name, v, isConst, pos);
}

const UndefinedStructType *UndefinedStructType::GetAsConstType() const {
    if (isConst)
        return this;
    return new UndefinedStructType(name, variability, true, pos);
}

const UndefinedStructType *UndefinedStructType::GetAsNonConstType() const {
    if (isConst == false)
        return this;
    return new UndefinedStructType(name, variability, false, pos);
}

std::string UndefinedStructType::GetString() const {
    std::string ret;
    if (isConst)
        ret += "const ";
    ret += variability.GetString();
    ret += " struct ";
    ret += name;
    return ret;
}

std::string UndefinedStructType::Mangle() const { return lMangleStruct(variability, isConst, name); }

std::string UndefinedStructType::GetCDeclaration(const std::string &n) const {
    std::string ret;
    if (isConst)
        ret += "const ";
    ret += std::string("struct ") + name;
    if (lShouldPrintName(n))
        ret += std::string(" ") + n;
    return ret;
}

llvm::Type *UndefinedStructType::LLVMType(llvm::LLVMContext *ctx) const {
    Assert(variability != Variability::Unbound);
    std::string mname = lMangleStructName(name, variability);
    if (m->structTypeMap.find(mname) == m->structTypeMap.end()) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return m->structTypeMap[mname];
}

llvm::DIType *UndefinedStructType::GetDIType(llvm::DIScope *scope) const {
    llvm::DIFile *diFile = pos.GetDIFile();
    llvm::DINamespace *diSpace = pos.GetDINamespace();
    llvm::DINodeArray elements;
    return m->diBuilder->createStructType(diSpace, GetString(), diFile,
                                          pos.first_line,         // Line number
                                          0,                      // Size
                                          0,                      // Align
                                          llvm::DINode::FlagZero, // Flags
                                          NULL, elements);
}

///////////////////////////////////////////////////////////////////////////
// ReferenceType

ReferenceType::ReferenceType(const Type *t, AddressSpace as) : Type(REFERENCE_TYPE), targetType(t), addrSpace(as) {
    asOtherConstType = NULL;
}

Variability ReferenceType::GetVariability() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return Variability(Variability::Unbound);
    }
    return targetType->GetVariability();
}

bool ReferenceType::IsBoolType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsBoolType();
}

bool ReferenceType::IsFloatType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsFloatType();
}

bool ReferenceType::IsIntType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsIntType();
}

bool ReferenceType::IsUnsignedType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsUnsignedType();
}

bool ReferenceType::IsConstType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsConstType();
}

const Type *ReferenceType::GetReferenceTarget() const { return targetType; }

const Type *ReferenceType::GetBaseType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return targetType->GetBaseType();
}

const ReferenceType *ReferenceType::GetAsVaryingType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    if (IsVaryingType())
        return this;
    return new ReferenceType(targetType->GetAsVaryingType());
}

const ReferenceType *ReferenceType::GetAsUniformType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    if (IsUniformType())
        return this;
    return new ReferenceType(targetType->GetAsUniformType());
}

const ReferenceType *ReferenceType::GetAsUnboundVariabilityType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    if (HasUnboundVariability())
        return this;
    return new ReferenceType(targetType->GetAsUnboundVariabilityType());
}

const Type *ReferenceType::GetAsSOAType(int width) const {
    // FIXME: is this right?
    return new ArrayType(this, width);
}

const ReferenceType *ReferenceType::ResolveUnboundVariability(Variability v) const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    return new ReferenceType(targetType->ResolveUnboundVariability(v));
}

const ReferenceType *ReferenceType::GetAsConstType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    if (IsConstType())
        return this;

    if (asOtherConstType == NULL) {
        asOtherConstType = new ReferenceType(targetType->GetAsConstType());
        asOtherConstType->asOtherConstType = this;
    }
    return asOtherConstType;
}

const ReferenceType *ReferenceType::GetAsNonConstType() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    if (!IsConstType())
        return this;

    if (asOtherConstType == NULL) {
        asOtherConstType = new ReferenceType(targetType->GetAsNonConstType());
        asOtherConstType->asOtherConstType = this;
    }
    return asOtherConstType;
}

const ReferenceType *ReferenceType::GetWithAddrSpace(AddressSpace as) const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    if (addrSpace == as)
        return this;

    return new ReferenceType(targetType, as);
}

std::string ReferenceType::GetString() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret = targetType->GetString();

    ret += std::string(" &");
    return ret;
}

std::string ReferenceType::Mangle() const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }
    std::string ret;
    ret += std::string("REF") + targetType->Mangle();
    return ret;
}

std::string ReferenceType::GetCDeclaration(const std::string &name) const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return "";
    }

    const ArrayType *at = CastType<ArrayType>(targetType);
    if (at != NULL) {
        if (at->GetElementCount() == 0) {
            // emit unsized arrays as pointers to the base type..
            std::string ret;
            ret += at->GetElementType()->GetAsNonConstType()->GetCDeclaration("") + std::string(" *");
            if (lShouldPrintName(name))
                ret += name;
            return ret;
        } else
            // otherwise forget about the reference part if it's an
            // array since C already passes arrays by reference...
            return targetType->GetCDeclaration(name);
    } else {
        std::string ret;
        ret += targetType->GetCDeclaration("") + std::string(" &");
        if (lShouldPrintName(name))
            ret += name;
        return ret;
    }
}

llvm::Type *ReferenceType::LLVMType(llvm::LLVMContext *ctx) const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    llvm::Type *t = targetType->LLVMStorageType(ctx);
    if (t == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    return llvm::PointerType::get(t, (unsigned)addrSpace);
}

llvm::DIType *ReferenceType::GetDIType(llvm::DIScope *scope) const {
    if (targetType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    llvm::DIType *diTargetType = targetType->GetDIType(scope);
    return m->diBuilder->createReferenceType(llvm::dwarf::DW_TAG_reference_type, diTargetType);
}

///////////////////////////////////////////////////////////////////////////
// FunctionType

FunctionType::FunctionType(const Type *r, const llvm::SmallVector<const Type *, 8> &a, SourcePos p)
    : Type(FUNCTION_TYPE), isTask(false), isExported(false), isExternC(false), isUnmasked(false), returnType(r),
      paramTypes(a), paramNames(llvm::SmallVector<std::string, 8>(a.size(), "")),
      paramDefaults(llvm::SmallVector<Expr *, 8>(a.size(), NULL)),
      paramPositions(llvm::SmallVector<SourcePos, 8>(a.size(), p)) {
    Assert(returnType != NULL);
    isSafe = false;
    costOverride = -1;
}

FunctionType::FunctionType(const Type *r, const llvm::SmallVector<const Type *, 8> &a,
                           const llvm::SmallVector<std::string, 8> &an, const llvm::SmallVector<Expr *, 8> &ad,
                           const llvm::SmallVector<SourcePos, 8> &ap, bool it, bool is, bool ec, bool ium)
    : Type(FUNCTION_TYPE), isTask(it), isExported(is), isExternC(ec), isUnmasked(ium), returnType(r), paramTypes(a),
      paramNames(an), paramDefaults(ad), paramPositions(ap) {
    Assert(paramTypes.size() == paramNames.size() && paramNames.size() == paramDefaults.size() &&
           paramDefaults.size() == paramPositions.size());
    Assert(returnType != NULL);
    isSafe = false;
    costOverride = -1;
}

Variability FunctionType::GetVariability() const { return Variability(Variability::Uniform); }

bool FunctionType::IsFloatType() const { return false; }

bool FunctionType::IsIntType() const { return false; }

bool FunctionType::IsBoolType() const { return false; }

bool FunctionType::IsUnsignedType() const { return false; }

bool FunctionType::IsConstType() const { return false; }

bool FunctionType::IsISPCKernel() const { return g->target->isXeTarget() && isTask; }

bool FunctionType::IsISPCExternal() const { return g->target->isXeTarget() && (isExported || isExternC); }

const Type *FunctionType::GetBaseType() const {
    FATAL("FunctionType::GetBaseType() shouldn't be called");
    return NULL;
}

const Type *FunctionType::GetAsVaryingType() const {
    FATAL("FunctionType::GetAsVaryingType shouldn't be called");
    return NULL;
}

const Type *FunctionType::GetAsUniformType() const {
    FATAL("FunctionType::GetAsUniformType shouldn't be called");
    return NULL;
}

const Type *FunctionType::GetAsUnboundVariabilityType() const {
    FATAL("FunctionType::GetAsUnboundVariabilityType shouldn't be called");
    return NULL;
}

const Type *FunctionType::GetAsSOAType(int width) const {
    FATAL("FunctionType::GetAsSOAType() shouldn't be called");
    return NULL;
}

const FunctionType *FunctionType::ResolveUnboundVariability(Variability v) const {
    if (returnType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }
    const Type *rt = returnType->ResolveUnboundVariability(v);

    llvm::SmallVector<const Type *, 8> pt;
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == NULL) {
            Assert(m->errorCount > 0);
            return NULL;
        }
        pt.push_back(paramTypes[i]->ResolveUnboundVariability(v));
    }

    FunctionType *ret =
        new FunctionType(rt, pt, paramNames, paramDefaults, paramPositions, isTask, isExported, isExternC, isUnmasked);
    ret->isSafe = isSafe;
    ret->costOverride = costOverride;

    return ret;
}

const Type *FunctionType::GetAsConstType() const { return this; }

const Type *FunctionType::GetAsNonConstType() const { return this; }

std::string FunctionType::GetString() const {
    std::string ret = GetReturnTypeString();
    ret += "(";
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == NULL)
            ret += "/* ERROR */";
        else
            ret += paramTypes[i]->GetString();

        if (i != paramTypes.size() - 1)
            ret += ", ";
    }
    ret += ")";
    return ret;
}

std::string FunctionType::Mangle() const {
    std::string ret = "___";
    if (isUnmasked)
        ret += "UM_";

    for (unsigned int i = 0; i < paramTypes.size(); ++i)
        if (paramTypes[i] == NULL)
            Assert(m->errorCount > 0);
        else
            ret += paramTypes[i]->Mangle();

    return ret;
}

std::string FunctionType::GetCDeclaration(const std::string &fname) const {
    std::string ret;
    ret += returnType->GetCDeclaration("");
    ret += " ";
    ret += fname;
    ret += "(";
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        const Type *type = paramTypes[i];

        // Convert pointers to arrays to unsized arrays, which are more clear
        // to print out for multidimensional arrays (i.e. "float foo[][4] "
        // versus "float (foo *)[4]").
        const PointerType *pt = CastType<PointerType>(type);
        if (pt != NULL && CastType<ArrayType>(pt->GetBaseType()) != NULL) {
            type = new ArrayType(pt->GetBaseType(), 0);
        }

        if (paramNames[i] != "")
            ret += type->GetCDeclaration(paramNames[i]);
        else
            ret += type->GetString();
        if (i != paramTypes.size() - 1)
            ret += ", ";
    }
    ret += ")";
    return ret;
}

std::string FunctionType::GetCDeclarationForDispatch(const std::string &fname) const {
    std::string ret;
    ret += returnType->GetCDeclaration("");
    ret += " ";
    ret += fname;
    ret += "(";
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        const Type *type = paramTypes[i];

        // Convert pointers to arrays to unsized arrays, which are more clear
        // to print out for multidimensional arrays (i.e. "float foo[][4] "
        // versus "float (foo *)[4]").
        const PointerType *pt = CastType<PointerType>(type);
        if (pt != NULL && CastType<ArrayType>(pt->GetBaseType()) != NULL) {
            type = new ArrayType(pt->GetBaseType(), 0);
        }

        // Change pointers to varying thingies to void *
        if (pt != NULL && pt->GetBaseType()->IsVaryingType()) {
            PointerType *t = PointerType::Void;

            if (paramNames[i] != "")
                ret += t->GetCDeclaration(paramNames[i]);
            else
                ret += t->GetString();
        } else {
            if (paramNames[i] != "")
                ret += type->GetCDeclaration(paramNames[i]);
            else
                ret += type->GetString();
        }
        if (i != paramTypes.size() - 1)
            ret += ", ";
    }
    ret += ")";
    return ret;
}

llvm::Type *FunctionType::LLVMType(llvm::LLVMContext *ctx) const {
    FATAL("FunctionType::LLVMType() shouldn't be called");
    return NULL;
}

llvm::DIType *FunctionType::GetDIType(llvm::DIScope *scope) const {

    std::vector<llvm::Metadata *> retArgTypes;
    retArgTypes.push_back(returnType->GetDIType(scope));
    for (int i = 0; i < GetNumParameters(); ++i) {
        const Type *t = GetParameterType(i);
        if (t == NULL)

            return NULL;
        retArgTypes.push_back(t->GetDIType(scope));
    }

    llvm::DITypeRefArray retArgTypesArray = m->diBuilder->getOrCreateTypeArray(retArgTypes);
    llvm::DIType *diType = m->diBuilder->createSubroutineType(retArgTypesArray);
    return diType;
}

const std::string FunctionType::GetReturnTypeString() const {
    if (returnType == NULL)
        return "/* ERROR */";

    std::string ret;
    if (isTask)
        ret += "task ";
    if (isExported)
        ret += "export ";
    if (isExternC)
        ret += "extern \"C\" ";
    if (isUnmasked)
        ret += "unmasked ";
    if (isSafe)
        ret += "/*safe*/ ";
    if (costOverride > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "/*cost=%d*/ ", costOverride);
        ret += buf;
    }

    return ret + returnType->GetString();
}

llvm::FunctionType *FunctionType::LLVMFunctionType(llvm::LLVMContext *ctx, bool removeMask) const {
    if (!g->target->isXeTarget() && isTask == true) {
        Assert(removeMask == false);
    }

    // Get the LLVM Type *s for the function arguments
    std::vector<llvm::Type *> llvmArgTypes;
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == NULL) {
            Assert(m->errorCount > 0);
            return NULL;
        }
        Assert(paramTypes[i]->IsVoidType() == false);

        const Type *argType = paramTypes[i];

        // We should cast pointers to generic address spaces
        // for ISPCExtenal functions (not-masked version) on Xe target
        llvm::Type *castedArgType = argType->LLVMType(ctx);

        if (IsISPCExternal() && removeMask) {
            if (argType->IsPointerType()) {
                const PointerType *argPtr =
                    (CastType<PointerType>(argType))->GetWithAddrSpace(AddressSpace::ispc_generic);
                castedArgType = argPtr->LLVMType(ctx);
            } else if (argType->IsReferenceType()) {
                const ReferenceType *refPtr =
                    (CastType<ReferenceType>(argType))->GetWithAddrSpace(AddressSpace::ispc_generic);
                castedArgType = refPtr->LLVMType(ctx);
            }
        }

        if (castedArgType == NULL) {
            Assert(m->errorCount > 0);
            return NULL;
        }
        llvmArgTypes.push_back(castedArgType);
    }

    // And add the function mask, if asked for
    if (!(removeMask || isUnmasked || IsISPCKernel())) {
        llvmArgTypes.push_back(LLVMTypes::MaskType);
    }

    std::vector<llvm::Type *> callTypes;
    if (isTask && (!g->target->isXeTarget())) {
        // Tasks take three arguments: a pointer to a struct that holds the
        // actual task arguments, the thread index, and the total number of
        // threads the tasks system has running.  (Task arguments are
        // marshalled in a struct so that it's easy to allocate space to
        // hold them until the task actually runs.)
        llvm::Type *st = llvm::StructType::get(*ctx, llvmArgTypes);
        callTypes.push_back(llvm::PointerType::getUnqual(st));
        callTypes.push_back(LLVMTypes::Int32Type); // threadIndex
        callTypes.push_back(LLVMTypes::Int32Type); // threadCount
        callTypes.push_back(LLVMTypes::Int32Type); // taskIndex
        callTypes.push_back(LLVMTypes::Int32Type); // taskCount
        callTypes.push_back(LLVMTypes::Int32Type); // taskIndex0
        callTypes.push_back(LLVMTypes::Int32Type); // taskIndex1
        callTypes.push_back(LLVMTypes::Int32Type); // taskIndex2
        callTypes.push_back(LLVMTypes::Int32Type); // taskCount0
        callTypes.push_back(LLVMTypes::Int32Type); // taskCount1
        callTypes.push_back(LLVMTypes::Int32Type); // taskCount2
    } else {
        // Otherwise we already have the types of the arguments
        callTypes = llvmArgTypes;
    }

    if (returnType == NULL) {
        Assert(m->errorCount > 0);
        return NULL;
    }

    const Type *retType = returnType;

    llvm::Type *llvmReturnType = retType->LLVMType(g->ctx);
    if (llvmReturnType == NULL)
        return NULL;
    return llvm::FunctionType::get(llvmReturnType, callTypes, false);
}

const Type *FunctionType::GetParameterType(int i) const {
    Assert(i < (int)paramTypes.size());
    return paramTypes[i];
}

Expr *FunctionType::GetParameterDefault(int i) const {
    Assert(i < (int)paramDefaults.size());
    return paramDefaults[i];
}

const SourcePos &FunctionType::GetParameterSourcePos(int i) const {
    Assert(i < (int)paramPositions.size());
    return paramPositions[i];
}

const std::string &FunctionType::GetParameterName(int i) const {
    Assert(i < (int)paramNames.size());
    return paramNames[i];
}

bool FunctionType::RequiresAddrSpaceCasts(const llvm::Function *func) const {
    return IsISPCExternal() && func->getCallingConv() == llvm::CallingConv::SPIR_FUNC;
}

///////////////////////////////////////////////////////////////////////////
// Type

const Type *Type::GetReferenceTarget() const {
    // only ReferenceType needs to override this method
    return this;
}

const Type *Type::GetAsUnsignedType() const {
    // For many types, this doesn't make any sesne
    return NULL;
}

/** Given an atomic or vector type, return a vector type of the given
    vecSize.  Issue an error if given a vector type that isn't already that
    size.
 */
static const Type *lVectorConvert(const Type *type, SourcePos pos, const char *reason, int vecSize) {
    const VectorType *vt = CastType<VectorType>(type);
    if (vt) {
        if (vt->GetElementCount() != vecSize) {
            Error(pos,
                  "Implicit conversion between from vector type "
                  "\"%s\" to vector type of length %d for %s is not possible.",
                  type->GetString().c_str(), vecSize, reason);
            return NULL;
        }
        return vt;
    } else {
        const AtomicType *at = CastType<AtomicType>(type);
        if (!at) {
            Error(pos,
                  "Non-atomic type \"%s\" can't be converted to vector type "
                  "for %s.",
                  type->GetString().c_str(), reason);
            return NULL;
        }
        return new VectorType(at, vecSize);
    }
}

const Type *Type::MoreGeneralType(const Type *t0, const Type *t1, SourcePos pos, const char *reason, bool forceVarying,
                                  int vecSize) {
    Assert(reason != NULL);

    // First, if one or both types are function types, convert them to
    // pointer to function types and then try again.
    if (CastType<FunctionType>(t0) || CastType<FunctionType>(t1)) {
        if (CastType<FunctionType>(t0))
            t0 = PointerType::GetUniform(t0);
        if (CastType<FunctionType>(t1))
            t1 = PointerType::GetUniform(t1);
        return MoreGeneralType(t0, t1, pos, reason, forceVarying, vecSize);
    }

    // First, if we need to go varying, promote both of the types to be
    // varying.
    if (t0->IsVaryingType() || t1->IsVaryingType() || forceVarying) {
        t0 = t0->GetAsVaryingType();
        t1 = t1->GetAsVaryingType();
    }

    // And similarly, promote them both to vectors if the caller requested
    // a particular vector size
    if (vecSize > 0) {
        t0 = lVectorConvert(t0, pos, reason, vecSize);
        t1 = lVectorConvert(t1, pos, reason, vecSize);
        if (!t0 || !t1)
            return NULL;
    }

    // Are they both the same type?  If so, we're done, QED.
    if (Type::Equal(t0, t1))
        return t0;

    // If they're function types, it's hopeless if they didn't match in the
    // Type::Equal() call above.  Fail here so that we don't get into
    // trouble calling GetAsConstType()...
    if (CastType<FunctionType>(t0) || CastType<FunctionType>(t1)) {
        Error(pos, "Incompatible function types \"%s\" and \"%s\" in %s.", t0->GetString().c_str(),
              t1->GetString().c_str(), reason);
        return NULL;
    }

    // Not the same types, but only a const/non-const difference?  Return
    // the non-const type as the more general one.
    if (Type::EqualIgnoringConst(t0, t1))
        return t0->GetAsNonConstType();

    const PointerType *pt0 = CastType<PointerType>(t0);
    const PointerType *pt1 = CastType<PointerType>(t1);
    if (pt0 != NULL && pt1 != NULL) {
        if (PointerType::IsVoidPointer(pt0))
            return pt1;
        else if (PointerType::IsVoidPointer(pt1))
            return pt0;
        else {
            Error(pos,
                  "Conversion between incompatible pointer types \"%s\" "
                  "and \"%s\" isn't possible.",
                  t0->GetString().c_str(), t1->GetString().c_str());
            return NULL;
        }
    }

    const VectorType *vt0 = CastType<VectorType>(t0->GetReferenceTarget());
    const VectorType *vt1 = CastType<VectorType>(t1->GetReferenceTarget());
    if (vt0 && vt1) {
        // both are vectors; convert their base types and make a new vector
        // type, as long as their lengths match
        if (vt0->GetElementCount() != vt1->GetElementCount()) {
            Error(pos,
                  "Implicit conversion between differently sized vector types "
                  "(%s, %s) for %s is not possible.",
                  t0->GetString().c_str(), t1->GetString().c_str(), reason);
            return NULL;
        }
        const Type *t = MoreGeneralType(vt0->GetElementType(), vt1->GetElementType(), pos, reason, forceVarying);
        if (!t)
            return NULL;

        // The 'more general' version of the two vector element types must
        // be an AtomicType (that's all that vectors can hold...)
        const AtomicType *at = CastType<AtomicType>(t);
        Assert(at != NULL);

        return new VectorType(at, vt0->GetElementCount());
    } else if (vt0) {
        // If one type is a vector type but the other isn't, see if we can
        // promote the other one to a vector type.  This will fail and
        // return NULL if t1 is e.g. an array type and it's illegal to have
        // a vector of it..
        const Type *t = MoreGeneralType(vt0->GetElementType(), t1, pos, reason, forceVarying);
        if (!t)
            return NULL;

        const AtomicType *at = CastType<AtomicType>(t);
        Assert(at != NULL);
        return new VectorType(at, vt0->GetElementCount());
    } else if (vt1) {
        // As in the above case, see if we can promote t0 to make a vector
        // that matches vt1.
        const Type *t = MoreGeneralType(t0, vt1->GetElementType(), pos, reason, forceVarying);
        if (!t)
            return NULL;

        const AtomicType *at = CastType<AtomicType>(t);
        Assert(at != NULL);
        return new VectorType(at, vt1->GetElementCount());
    }

    // TODO: what do we need to do about references here, if anything??

    const AtomicType *at0 = CastType<AtomicType>(t0->GetReferenceTarget());
    const AtomicType *at1 = CastType<AtomicType>(t1->GetReferenceTarget());

    const EnumType *et0 = CastType<EnumType>(t0->GetReferenceTarget());
    const EnumType *et1 = CastType<EnumType>(t1->GetReferenceTarget());
    if (et0 != NULL && et1 != NULL) {
        // Two different enum types -> make them uint32s...
        Assert(et0->IsVaryingType() == et1->IsVaryingType());
        return et0->IsVaryingType() ? AtomicType::VaryingUInt32 : AtomicType::UniformUInt32;
    } else if (et0 != NULL) {
        if (at1 != NULL)
            // Enum type and atomic type -> convert the enum to the atomic type
            // TODO: should we return uint32 here, unless the atomic type is
            // a 64-bit atomic type, in which case we return that?
            return at1;
        else {
            Error(pos,
                  "Implicit conversion from enum type \"%s\" to "
                  "non-atomic type \"%s\" for %s not possible.",
                  t0->GetString().c_str(), t1->GetString().c_str(), reason);
            return NULL;
        }
    } else if (et1 != NULL) {
        if (at0 != NULL)
            // Enum type and atomic type; see TODO above here as well...
            return at0;
        else {
            Error(pos,
                  "Implicit conversion from enum type \"%s\" to "
                  "non-atomic type \"%s\" for %s not possible.",
                  t1->GetString().c_str(), t0->GetString().c_str(), reason);
            return NULL;
        }
    }

    // Now all we can do is promote atomic types...
    if (at0 == NULL || at1 == NULL) {
        Assert(reason != NULL);
        Error(pos, "Implicit conversion from type \"%s\" to \"%s\" for %s not possible.", t0->GetString().c_str(),
              t1->GetString().c_str(), reason);
        return NULL;
    }

    // Finally, to determine which of the two atomic types is more general,
    // use the ordering of entries in the AtomicType::BasicType enumerator.
    return (int(at0->basicType) >= int(at1->basicType)) ? at0 : at1;
}

bool Type::IsBasicType(const Type *type) {
    return (CastType<AtomicType>(type) != NULL || CastType<EnumType>(type) != NULL ||
            CastType<PointerType>(type) != NULL);
}

static bool lCheckTypeEquality(const Type *a, const Type *b, bool ignoreConst) {
    if (a == NULL || b == NULL)
        return false;

    if (ignoreConst == false && a->IsConstType() != b->IsConstType())
        return false;

    const AtomicType *ata = CastType<AtomicType>(a);
    const AtomicType *atb = CastType<AtomicType>(b);
    if (ata != NULL && atb != NULL) {
        return ((ata->basicType == atb->basicType) && (ata->GetVariability() == atb->GetVariability()));
    }

    // For all of the other types, we need to see if we have the same two
    // general types.  If so, then we dig into the details of the type and
    // see if all of the relevant bits are equal...
    const EnumType *eta = CastType<EnumType>(a);
    const EnumType *etb = CastType<EnumType>(b);
    if (eta != NULL && etb != NULL)
        // Kind of goofy, but this sufficies to check
        return (eta->pos == etb->pos && eta->GetVariability() == etb->GetVariability());

    const ArrayType *arta = CastType<ArrayType>(a);
    const ArrayType *artb = CastType<ArrayType>(b);
    if (arta != NULL && artb != NULL)
        return (arta->GetElementCount() == artb->GetElementCount() &&
                lCheckTypeEquality(arta->GetElementType(), artb->GetElementType(), ignoreConst));

    const VectorType *vta = CastType<VectorType>(a);
    const VectorType *vtb = CastType<VectorType>(b);
    if (vta != NULL && vtb != NULL)
        return (vta->GetElementCount() == vtb->GetElementCount() &&
                lCheckTypeEquality(vta->GetElementType(), vtb->GetElementType(), ignoreConst));

    const StructType *sta = CastType<StructType>(a);
    const StructType *stb = CastType<StructType>(b);
    const UndefinedStructType *usta = CastType<UndefinedStructType>(a);
    const UndefinedStructType *ustb = CastType<UndefinedStructType>(b);
    if ((sta != NULL || usta != NULL) && (stb != NULL || ustb != NULL)) {
        // Report both defuned and undefined structs as equal if their
        // names are the same.
        if (a->GetVariability() != b->GetVariability())
            return false;

        const std::string &namea = sta ? sta->GetStructName() : usta->GetStructName();
        const std::string &nameb = stb ? stb->GetStructName() : ustb->GetStructName();
        return (namea == nameb);
    }

    const PointerType *pta = CastType<PointerType>(a);
    const PointerType *ptb = CastType<PointerType>(b);
    if (pta != NULL && ptb != NULL)
        return (pta->IsUniformType() == ptb->IsUniformType() && pta->IsSlice() == ptb->IsSlice() &&
                pta->IsFrozenSlice() == ptb->IsFrozenSlice() &&
                lCheckTypeEquality(pta->GetBaseType(), ptb->GetBaseType(), ignoreConst));

    const ReferenceType *rta = CastType<ReferenceType>(a);
    const ReferenceType *rtb = CastType<ReferenceType>(b);
    if (rta != NULL && rtb != NULL)
        return (lCheckTypeEquality(rta->GetReferenceTarget(), rtb->GetReferenceTarget(), ignoreConst));

    const FunctionType *fta = CastType<FunctionType>(a);
    const FunctionType *ftb = CastType<FunctionType>(b);
    if (fta != NULL && ftb != NULL) {
        // Both the return types and all of the argument types must match
        // for function types to match
        if (!lCheckTypeEquality(fta->GetReturnType(), ftb->GetReturnType(), ignoreConst))
            return false;

        if (fta->isTask != ftb->isTask || fta->isExported != ftb->isExported || fta->isExternC != ftb->isExternC ||
            fta->isUnmasked != ftb->isUnmasked)
            return false;

        if (fta->GetNumParameters() != ftb->GetNumParameters())
            return false;

        for (int i = 0; i < fta->GetNumParameters(); ++i)
            if (!lCheckTypeEquality(fta->GetParameterType(i), ftb->GetParameterType(i), ignoreConst))
                return false;

        return true;
    }

    return false;
}

bool Type::Equal(const Type *a, const Type *b) { return lCheckTypeEquality(a, b, false); }

bool Type::EqualIgnoringConst(const Type *a, const Type *b) { return lCheckTypeEquality(a, b, true); }
