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

/** @file type.cpp
    @brief Definitions for classes related to type representation
*/

#include "type.h"
#include "expr.h"
#include "util.h"
#include "sym.h"
#include "llvmutil.h"
#include "module.h"

#include <stdio.h>
#include <llvm/Value.h>
#include <llvm/Module.h>
#ifndef LLVM_2_8
#include <llvm/Analysis/DIBuilder.h>
#endif
#include <llvm/Analysis/DebugInfo.h>
#include <llvm/Support/Dwarf.h>


/** Utility routine used in code that prints out declarations; returns true
    if the given name should be printed, false otherwise.  This allows us
    to omit the names for various internal things (whose names start with
    double underscores) and emit anonymous declarations for them instead.
 */

static bool
lShouldPrintName(const std::string &name) {
    if (name.size() == 0)
        return false;
    else if (name[0] != '_')
        return true;
    else
        return (name.size() == 1) || (name[1] != '_');
}


///////////////////////////////////////////////////////////////////////////
// AtomicType

const AtomicType *AtomicType::UniformBool = new AtomicType(TYPE_BOOL, true, false);
const AtomicType *AtomicType::VaryingBool = new AtomicType(TYPE_BOOL, false, false);
const AtomicType *AtomicType::UniformInt8 = new AtomicType(TYPE_INT8, true, false);
const AtomicType *AtomicType::VaryingInt8 = new AtomicType(TYPE_INT8, false, false);
const AtomicType *AtomicType::UniformUInt8 = new AtomicType(TYPE_UINT8, true, false);
const AtomicType *AtomicType::VaryingUInt8 = new AtomicType(TYPE_UINT8, false, false);
const AtomicType *AtomicType::UniformInt16 = new AtomicType(TYPE_INT16, true, false);
const AtomicType *AtomicType::VaryingInt16 = new AtomicType(TYPE_INT16, false, false);
const AtomicType *AtomicType::UniformUInt16 = new AtomicType(TYPE_UINT16, true, false);
const AtomicType *AtomicType::VaryingUInt16 = new AtomicType(TYPE_UINT16, false, false);
const AtomicType *AtomicType::UniformInt32 = new AtomicType(TYPE_INT32, true, false);
const AtomicType *AtomicType::VaryingInt32 = new AtomicType(TYPE_INT32, false, false);
const AtomicType *AtomicType::UniformUInt32 = new AtomicType(TYPE_UINT32, true, false);
const AtomicType *AtomicType::VaryingUInt32 = new AtomicType(TYPE_UINT32, false, false);
const AtomicType *AtomicType::UniformFloat = new AtomicType(TYPE_FLOAT, true, false);
const AtomicType *AtomicType::VaryingFloat = new AtomicType(TYPE_FLOAT, false, false);
const AtomicType *AtomicType::UniformInt64 = new AtomicType(TYPE_INT64, true, false);
const AtomicType *AtomicType::VaryingInt64 = new AtomicType(TYPE_INT64, false, false);
const AtomicType *AtomicType::UniformUInt64 = new AtomicType(TYPE_UINT64, true, false);
const AtomicType *AtomicType::VaryingUInt64 = new AtomicType(TYPE_UINT64, false, false);
const AtomicType *AtomicType::UniformDouble = new AtomicType(TYPE_DOUBLE, true, false);
const AtomicType *AtomicType::VaryingDouble = new AtomicType(TYPE_DOUBLE, false, false);

const AtomicType *AtomicType::UniformConstBool = new AtomicType(TYPE_BOOL, true, true);
const AtomicType *AtomicType::VaryingConstBool = new AtomicType(TYPE_BOOL, false, true);
const AtomicType *AtomicType::UniformConstInt8 = new AtomicType(TYPE_INT8, true, true);
const AtomicType *AtomicType::VaryingConstInt8 = new AtomicType(TYPE_INT8, false, true);
const AtomicType *AtomicType::UniformConstUInt8 = new AtomicType(TYPE_UINT8, true, true);
const AtomicType *AtomicType::VaryingConstUInt8 = new AtomicType(TYPE_UINT8, false, true);
const AtomicType *AtomicType::UniformConstInt16 = new AtomicType(TYPE_INT16, true, true);
const AtomicType *AtomicType::VaryingConstInt16 = new AtomicType(TYPE_INT16, false, true);
const AtomicType *AtomicType::UniformConstUInt16 = new AtomicType(TYPE_UINT16, true, true);
const AtomicType *AtomicType::VaryingConstUInt16 = new AtomicType(TYPE_UINT16, false, true);
const AtomicType *AtomicType::UniformConstInt32 = new AtomicType(TYPE_INT32, true, true);
const AtomicType *AtomicType::VaryingConstInt32 = new AtomicType(TYPE_INT32, false, true);
const AtomicType *AtomicType::UniformConstUInt32 = new AtomicType(TYPE_UINT32, true, true);
const AtomicType *AtomicType::VaryingConstUInt32 = new AtomicType(TYPE_UINT32, false, true);
const AtomicType *AtomicType::UniformConstFloat = new AtomicType(TYPE_FLOAT, true, true);
const AtomicType *AtomicType::VaryingConstFloat = new AtomicType(TYPE_FLOAT, false, true);
const AtomicType *AtomicType::UniformConstInt64 = new AtomicType(TYPE_INT64, true, true);
const AtomicType *AtomicType::VaryingConstInt64 = new AtomicType(TYPE_INT64, false, true);
const AtomicType *AtomicType::UniformConstUInt64 = new AtomicType(TYPE_UINT64, true, true);
const AtomicType *AtomicType::VaryingConstUInt64 = new AtomicType(TYPE_UINT64, false, true);
const AtomicType *AtomicType::UniformConstDouble = new AtomicType(TYPE_DOUBLE, true, true);
const AtomicType *AtomicType::VaryingConstDouble = new AtomicType(TYPE_DOUBLE, false, true);

const AtomicType *AtomicType::Void = new AtomicType(TYPE_VOID, true, false);


AtomicType::AtomicType(BasicType bt, bool iu, bool ic) 
    : basicType(bt), isUniform(iu), isConst(ic) {
}


bool
AtomicType::IsUniformType() const {
    return isUniform;
}


bool
AtomicType::IsFloatType() const {
    return (basicType == TYPE_FLOAT || basicType == TYPE_DOUBLE);
}


bool
AtomicType::IsIntType() const {
    return (basicType == TYPE_INT8  || basicType == TYPE_UINT8  ||
            basicType == TYPE_INT16 || basicType == TYPE_UINT16 ||
            basicType == TYPE_INT32 || basicType == TYPE_UINT32 ||
            basicType == TYPE_INT64 || basicType == TYPE_UINT64);
}


bool
AtomicType::IsUnsignedType() const {
    return (basicType == TYPE_UINT8  || basicType == TYPE_UINT16 ||
            basicType == TYPE_UINT32 || basicType == TYPE_UINT64);
}


bool
AtomicType::IsBoolType() const {
    return basicType == TYPE_BOOL;
}


bool
AtomicType::IsConstType() const { 
    return isConst; 
}


const AtomicType *
AtomicType::GetAsUnsignedType() const {
    if (IsUnsignedType()) 
        return this;

    if      (this == AtomicType::UniformInt8)       return AtomicType::UniformUInt8;
    else if (this == AtomicType::VaryingInt8)       return AtomicType::VaryingUInt8;
    else if (this == AtomicType::UniformInt16)      return AtomicType::UniformUInt16;
    else if (this == AtomicType::VaryingInt16)      return AtomicType::VaryingUInt16;
    else if (this == AtomicType::UniformInt32)      return AtomicType::UniformUInt32;
    else if (this == AtomicType::VaryingInt32)      return AtomicType::VaryingUInt32;
    else if (this == AtomicType::UniformInt64)      return AtomicType::UniformUInt64;
    else if (this == AtomicType::VaryingInt64)      return AtomicType::VaryingUInt64;
    else if (this == AtomicType::UniformConstInt8)  return AtomicType::UniformConstUInt8;
    else if (this == AtomicType::VaryingConstInt8)  return AtomicType::VaryingConstUInt8;
    else if (this == AtomicType::UniformConstInt16) return AtomicType::UniformConstUInt16;
    else if (this == AtomicType::VaryingConstInt16) return AtomicType::VaryingConstUInt16;
    else if (this == AtomicType::UniformConstInt32) return AtomicType::UniformConstUInt32;
    else if (this == AtomicType::VaryingConstInt32) return AtomicType::VaryingConstUInt32;
    else if (this == AtomicType::UniformConstInt64) return AtomicType::UniformConstUInt64;
    else if (this == AtomicType::VaryingConstInt64) return AtomicType::VaryingConstUInt64;
    else                                            return NULL;
}


const AtomicType *
AtomicType::GetAsConstType() const {
    if (this == AtomicType::Void) 
        return this;

    switch (basicType) {
    case TYPE_BOOL:    return isUniform ? UniformConstBool   : VaryingConstBool;
    case TYPE_INT8:    return isUniform ? UniformConstInt8   : VaryingConstInt8;
    case TYPE_UINT8:   return isUniform ? UniformConstUInt8  : VaryingConstUInt8;
    case TYPE_INT16:   return isUniform ? UniformConstInt16  : VaryingConstInt16;
    case TYPE_UINT16:  return isUniform ? UniformConstUInt16 : VaryingConstUInt16;
    case TYPE_INT32:   return isUniform ? UniformConstInt32  : VaryingConstInt32;
    case TYPE_UINT32:  return isUniform ? UniformConstUInt32 : VaryingConstUInt32;
    case TYPE_FLOAT:   return isUniform ? UniformConstFloat  : VaryingConstFloat;
    case TYPE_INT64:   return isUniform ? UniformConstInt64  : VaryingConstInt64;
    case TYPE_UINT64:  return isUniform ? UniformConstUInt64 : VaryingConstUInt64;
    case TYPE_DOUBLE:  return isUniform ? UniformConstDouble : VaryingConstDouble;
    default:
        FATAL("logic error in AtomicType::GetAsConstType()");
        return NULL;
    }
}


const AtomicType *
AtomicType::GetAsNonConstType() const {
    if (this == AtomicType::Void) 
        return this;

    switch (basicType) {
    case TYPE_BOOL:    return isUniform ? UniformBool   : VaryingBool;
    case TYPE_INT8:    return isUniform ? UniformInt8   : VaryingInt8;
    case TYPE_UINT8:   return isUniform ? UniformUInt8  : VaryingUInt8;
    case TYPE_INT16:   return isUniform ? UniformInt16  : VaryingInt16;
    case TYPE_UINT16:  return isUniform ? UniformUInt16 : VaryingUInt16;
    case TYPE_INT32:   return isUniform ? UniformInt32  : VaryingInt32;
    case TYPE_UINT32:  return isUniform ? UniformUInt32 : VaryingUInt32;
    case TYPE_FLOAT:   return isUniform ? UniformFloat  : VaryingFloat;
    case TYPE_INT64:   return isUniform ? UniformInt64  : VaryingInt64;
    case TYPE_UINT64:  return isUniform ? UniformUInt64 : VaryingUInt64;
    case TYPE_DOUBLE:  return isUniform ? UniformDouble : VaryingDouble;
    default:
        FATAL("logic error in AtomicType::GetAsNonConstType()");
        return NULL;
    }
}


const AtomicType *
AtomicType::GetBaseType() const {
    return this;
}


const AtomicType *
AtomicType::GetAsVaryingType() const {
    if (IsVaryingType()) 
        return this;

    switch (basicType) {
    case TYPE_VOID:   return this;
    case TYPE_BOOL:   return isConst ? VaryingConstBool   : VaryingBool;
    case TYPE_INT8:   return isConst ? VaryingConstInt8   : VaryingInt8;
    case TYPE_UINT8:  return isConst ? VaryingConstUInt8  : VaryingUInt8;
    case TYPE_INT16:  return isConst ? VaryingConstInt16  : VaryingInt16;
    case TYPE_UINT16: return isConst ? VaryingConstUInt16 : VaryingUInt16;
    case TYPE_INT32:  return isConst ? VaryingConstInt32  : VaryingInt32;
    case TYPE_UINT32: return isConst ? VaryingConstUInt32 : VaryingUInt32;
    case TYPE_FLOAT:  return isConst ? VaryingConstFloat  : VaryingFloat;
    case TYPE_INT64:  return isConst ? VaryingConstInt64  : VaryingInt64;
    case TYPE_UINT64: return isConst ? VaryingConstUInt64 : VaryingUInt64;
    case TYPE_DOUBLE: return isConst ? VaryingConstDouble : VaryingDouble;
    default:          FATAL("Logic error in AtomicType::GetAsVaryingType()");
    }
    return NULL;
}


const AtomicType *
AtomicType::GetAsUniformType() const {
    if (IsUniformType()) 
        return this;

    switch (basicType) {
    case TYPE_VOID:   return this;
    case TYPE_BOOL:   return isConst ? UniformConstBool   : UniformBool;
    case TYPE_INT8:   return isConst ? UniformConstInt8   : UniformInt8;
    case TYPE_UINT8:  return isConst ? UniformConstUInt8  : UniformUInt8;
    case TYPE_INT16:  return isConst ? UniformConstInt16  : UniformInt16;
    case TYPE_UINT16: return isConst ? UniformConstUInt16 : UniformUInt16;
    case TYPE_INT32:  return isConst ? UniformConstInt32  : UniformInt32;
    case TYPE_UINT32: return isConst ? UniformConstUInt32 : UniformUInt32;
    case TYPE_FLOAT:  return isConst ? UniformConstFloat  : UniformFloat;
    case TYPE_INT64:  return isConst ? UniformConstInt64  : UniformInt64;
    case TYPE_UINT64: return isConst ? UniformConstUInt64 : UniformUInt64;
    case TYPE_DOUBLE: return isConst ? UniformConstDouble : UniformDouble;
    default:          FATAL("Logic error in AtomicType::GetAsUniformType()");
    }
    return NULL;
}


const Type *
AtomicType::GetSOAType(int width) const {
    assert(width > 0);
    return new ArrayType(this, width);
}


std::string
AtomicType::GetString() const {
    std::string ret;
    if (basicType != TYPE_VOID) {
        if (isConst)   ret += "const ";
        if (isUniform) ret += "uniform ";
    }

    switch (basicType) {
    case TYPE_VOID:   ret += "void";            break;
    case TYPE_BOOL:   ret += "bool";            break;
    case TYPE_INT8:   ret += "int8";            break;
    case TYPE_UINT8:  ret += "unsigned int8";   break;
    case TYPE_INT16:  ret += "int16";           break;
    case TYPE_UINT16: ret += "unsigned int16";  break;
    case TYPE_INT32:  ret += "int32";           break;
    case TYPE_UINT32: ret += "unsigned int32";  break;
    case TYPE_FLOAT:  ret += "float";           break;
    case TYPE_INT64:  ret += "int64";           break;
    case TYPE_UINT64: ret += "unsigned int64";  break;
    case TYPE_DOUBLE: ret += "double";          break;
    default: FATAL("Logic error in AtomicType::GetString()");
    }
    return ret;
}


std::string
AtomicType::Mangle() const {
    std::string ret;
    if (isConst)   ret += "C";
    if (isUniform) ret += "U";

    switch (basicType) {
    case TYPE_VOID:   ret += "v"; break;
    case TYPE_BOOL:   ret += "b"; break;
    case TYPE_INT8:   ret += "t"; break;
    case TYPE_UINT8:  ret += "T"; break;
    case TYPE_INT16:  ret += "s"; break;
    case TYPE_UINT16: ret += "S"; break;
    case TYPE_INT32:  ret += "i"; break;
    case TYPE_UINT32: ret += "u"; break;
    case TYPE_FLOAT:  ret += "f"; break;
    case TYPE_INT64:  ret += "I"; break;
    case TYPE_UINT64: ret += "U"; break;
    case TYPE_DOUBLE: ret += "d"; break;
    default: FATAL("Logic error in AtomicType::Mangle()");
    }
    return ret;
}


std::string
AtomicType::GetCDeclaration(const std::string &name) const {
    std::string ret;
    assert(isUniform);
    if (isConst) ret += "const ";

    switch (basicType) {
    case TYPE_VOID:   ret += "void";     break;
    case TYPE_BOOL:   ret += "bool";     break;
    case TYPE_INT8:   ret += "int8_t";   break;
    case TYPE_UINT8:  ret += "uint8_t";  break;
    case TYPE_INT16:  ret += "int16_t";  break;
    case TYPE_UINT16: ret += "uint16_t"; break;
    case TYPE_INT32:  ret += "int32_t";  break;
    case TYPE_UINT32: ret += "uint32_t"; break;
    case TYPE_FLOAT:  ret += "float";    break;
    case TYPE_INT64:  ret += "int64_t";  break;
    case TYPE_UINT64: ret += "uint64_t"; break;
    case TYPE_DOUBLE: ret += "double";   break;
    default: FATAL("Logic error in AtomicType::GetCDeclaration()");
    }

    if (lShouldPrintName(name)) {
        ret += " ";
        ret += name;
    }
    return ret;
}


LLVM_TYPE_CONST llvm::Type *
AtomicType::LLVMType(llvm::LLVMContext *ctx) const {
    switch (basicType) {
    case TYPE_VOID:
        return llvm::Type::getVoidTy(*ctx);
    case TYPE_BOOL:
        return isUniform ? LLVMTypes::BoolType : LLVMTypes::BoolVectorType;
    case TYPE_INT8:
    case TYPE_UINT8:
        return isUniform ? LLVMTypes::Int8Type : LLVMTypes::Int8VectorType;
    case TYPE_INT16:
    case TYPE_UINT16:
        return isUniform ? LLVMTypes::Int16Type : LLVMTypes::Int16VectorType;
    case TYPE_INT32:
    case TYPE_UINT32:
        return isUniform ? LLVMTypes::Int32Type : LLVMTypes::Int32VectorType;
    case TYPE_FLOAT:
        return isUniform ? LLVMTypes::FloatType : LLVMTypes::FloatVectorType;
    case TYPE_INT64:
    case TYPE_UINT64:
        return isUniform ? LLVMTypes::Int64Type : LLVMTypes::Int64VectorType;
    case TYPE_DOUBLE:
        return isUniform ? LLVMTypes::DoubleType : LLVMTypes::DoubleVectorType;
    default:
        FATAL("logic error in AtomicType::LLVMType");
        return NULL;
    }
}


llvm::DIType
AtomicType::GetDIType(llvm::DIDescriptor scope) const {
#ifdef LLVM_2_8
    FATAL("debug info not supported in llvm 2.8");
    return llvm::DIType();
#else
    if (isUniform) {
        switch (basicType) {
        case TYPE_VOID:
            return llvm::DIType();
        case TYPE_BOOL:
            return m->diBuilder->createBasicType("bool", 32 /* size */, 32 /* align */,
                                                 llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_INT8:
            return m->diBuilder->createBasicType("int8", 8 /* size */, 8 /* align */,
                                                 llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT8:
            return m->diBuilder->createBasicType("uint8", 8 /* size */, 8 /* align */,
                                                 llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_INT16:
            return m->diBuilder->createBasicType("int16", 16 /* size */, 16 /* align */,
                                                 llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT16:
            return m->diBuilder->createBasicType("uint16", 16 /* size */, 16 /* align */,
                                                 llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_INT32:
            return m->diBuilder->createBasicType("int32", 32 /* size */, 32 /* align */,
                                                 llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT32:
            return m->diBuilder->createBasicType("uint32", 32 /* size */, 32 /* align */,
                                                 llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_FLOAT:
            return m->diBuilder->createBasicType("float", 32 /* size */, 32 /* align */,
                                                 llvm::dwarf::DW_ATE_float);
            break;
        case TYPE_DOUBLE:
            return m->diBuilder->createBasicType("double", 64 /* size */, 64 /* align */,
                                                 llvm::dwarf::DW_ATE_float);
            break;
        case TYPE_INT64:
            return m->diBuilder->createBasicType("int64", 64 /* size */, 64 /* align */,
                                                 llvm::dwarf::DW_ATE_signed);
            break;
        case TYPE_UINT64:
            return m->diBuilder->createBasicType("uint64", 64 /* size */, 64 /* align */,
                                                 llvm::dwarf::DW_ATE_unsigned);
            break;
        default:
            FATAL("unhandled basic type in AtomicType::GetDIType()");
            return llvm::DIType();
        }
    }
    else {
        llvm::DIType unifType = GetAsUniformType()->GetDIType(scope);
        llvm::Value *sub = m->diBuilder->getOrCreateSubrange(0, g->target.vectorWidth-1);
#ifdef LLVM_2_9
        llvm::Value *suba[] = { sub };
        llvm::DIArray subArray = m->diBuilder->getOrCreateArray(suba, 1);
#else
        llvm::DIArray subArray = m->diBuilder->getOrCreateArray(sub);
#endif // LLVM_2_9
        uint64_t size =  unifType.getSizeInBits()  * g->target.vectorWidth;
        uint64_t align = unifType.getAlignInBits() * g->target.vectorWidth;
        return m->diBuilder->createVectorType(size, align, unifType, subArray);
    }
#endif // LLVM_2_8
}


///////////////////////////////////////////////////////////////////////////
// EnumType

EnumType::EnumType(SourcePos p) 
    : pos(p) {
    //    name = "/* (anonymous) */";
    isConst = false;
    isUniform = false;
}


EnumType::EnumType(const char *n, SourcePos p) 
    : pos(p), name(n) {
    isConst = false;
    isUniform = false;
}


bool 
EnumType::IsUniformType() const {
    return isUniform;
}


bool 
EnumType::IsBoolType() const {
    return false;
}


bool 
EnumType::IsFloatType() const {
    return false;
}


bool 
EnumType::IsIntType() const {
    return true;
}


bool 
EnumType::IsUnsignedType() const {
    return true;
}


bool 
EnumType::IsConstType() const {
    return isConst;
}


const EnumType *
EnumType::GetBaseType() const {
    return this;
}


const EnumType *
EnumType::GetAsVaryingType() const {
    if (IsVaryingType())
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->isUniform = false;
        return enumType;
    }
}


const EnumType *
EnumType::GetAsUniformType() const {
    if (IsUniformType())
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->isUniform = true;
        return enumType;
    }
}


const Type *
EnumType::GetSOAType(int width) const {
    assert(width > 0);
    return new ArrayType(this, width);
}


const EnumType *
EnumType::GetAsConstType() const {
    if (isConst)
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->isConst = true;
        return enumType;
    }
}


const EnumType *
EnumType::GetAsNonConstType() const {
    if (!isConst)
        return this;
    else {
        EnumType *enumType = new EnumType(*this);
        enumType->isConst = false;
        return enumType;
    }
}


std::string 
EnumType::GetString() const {
    std::string ret;
    if (isConst) ret += "const ";
    if (isUniform) ret += "uniform ";
    ret += "enum ";
    if (name.size())
        ret += name;
    return ret;
}


std::string 
EnumType::Mangle() const {
    std::string ret = std::string("enum[") + name + std::string("]");
    return ret;
}


std::string 
EnumType::GetCDeclaration(const std::string &varName) const {
    std::string ret;
    if (isConst) ret += "const ";
    ret += "enum";
    if (name.size())
        ret += std::string(" ") + name;
    if (lShouldPrintName(varName)) {
        ret += " ";
        ret += varName;
    }
    return ret;
}


LLVM_TYPE_CONST llvm::Type *
EnumType::LLVMType(llvm::LLVMContext *ctx) const {
    return isUniform ? LLVMTypes::Int32Type : LLVMTypes::Int32VectorType;
}


llvm::DIType 
EnumType::GetDIType(llvm::DIDescriptor scope) const {
#ifdef LLVM_2_8
    FATAL("debug info not supported in llvm 2.8");
    return llvm::DIType();
#else
    std::vector<llvm::Value *> enumeratorDescriptors;
    for (unsigned int i = 0; i < enumerators.size(); ++i) {
        unsigned int enumeratorValue;
        assert(enumerators[i]->constValue != NULL);
        int count = enumerators[i]->constValue->AsUInt32(&enumeratorValue);
        assert(count == 1);

        llvm::Value *descriptor = 
            m->diBuilder->createEnumerator(enumerators[i]->name, enumeratorValue);
        enumeratorDescriptors.push_back(descriptor);
    }
#ifdef LLVM_2_9
    llvm::DIArray elementArray = 
        m->diBuilder->getOrCreateArray(&enumeratorDescriptors[0],
                                       enumeratorDescriptors.size());
#else
    llvm::DIArray elementArray = 
        m->diBuilder->getOrCreateArray(enumeratorDescriptors);
#endif

    llvm::DIFile diFile = pos.GetDIFile();
    llvm::DIType diType =
        m->diBuilder->createEnumerationType(scope, name, diFile, pos.first_line,
                                            32 /* size in bits */,
                                            32 /* align in bits */,
                                            elementArray);
    if (IsUniformType())
        return diType;

    llvm::Value *sub = m->diBuilder->getOrCreateSubrange(0, g->target.vectorWidth-1);
#ifdef LLVM_2_9
    llvm::Value *suba[] = { sub };
    llvm::DIArray subArray = m->diBuilder->getOrCreateArray(suba, 1);
#else
    llvm::DIArray subArray = m->diBuilder->getOrCreateArray(sub);
#endif // !LLVM_2_9
    uint64_t size =  diType.getSizeInBits()  * g->target.vectorWidth;
    uint64_t align = diType.getAlignInBits() * g->target.vectorWidth;
    return m->diBuilder->createVectorType(size, align, diType, subArray);
#endif // !LLVM_2_8
}


void
EnumType::SetEnumerators(const std::vector<Symbol *> &e) {
    enumerators = e;
}


int
EnumType::GetEnumeratorCount() const {
    return (int)enumerators.size();
}


const Symbol *
EnumType::GetEnumerator(int i) const {
    return enumerators[i];
}


///////////////////////////////////////////////////////////////////////////
// SequentialType

const Type *SequentialType::GetElementType(int index) const {
    return GetElementType();
}


///////////////////////////////////////////////////////////////////////////
// ArrayType

ArrayType::ArrayType(const Type *c, int a) 
    : child(c), numElements(a) {
    // 0 -> unsized array.
    assert(numElements >= 0);
}


LLVM_TYPE_CONST llvm::ArrayType *
ArrayType::LLVMType(llvm::LLVMContext *ctx) const {
    if (!child)
        return NULL;

    LLVM_TYPE_CONST llvm::Type *ct = child->LLVMType(ctx);
    if (!ct)
        return NULL;
    return llvm::ArrayType::get(ct, numElements);
}


bool
ArrayType::IsUniformType() const {
    return child->IsUniformType(); 
}


bool
ArrayType::IsFloatType() const {
    return false; 
}


bool
ArrayType::IsIntType() const {
    return false; 
}


bool
ArrayType::IsUnsignedType() const {
    return false; 
}


bool
ArrayType::IsBoolType() const {
    return false; 
}


bool
ArrayType::IsConstType() const {
    return child->IsConstType(); 
}


const Type *
ArrayType::GetBaseType() const {
    const Type *type = child;
    const ArrayType *at = dynamic_cast<const ArrayType *>(type);
    // Keep walking until we reach a child that isn't itself an array
    while (at) {
        type = at->child;
        at = dynamic_cast<const ArrayType *>(type);
    }
    return type;
}


const ArrayType *
ArrayType::GetAsVaryingType() const {
    return new ArrayType(child->GetAsVaryingType(), numElements);
}


const ArrayType *
ArrayType::GetAsUniformType() const {
    return new ArrayType(child->GetAsUniformType(), numElements);
}


const Type *
ArrayType::GetSOAType(int width) const {
    return new ArrayType(child->GetSOAType(width), numElements);
}


const ArrayType *
ArrayType::GetAsConstType() const {
    return new ArrayType(child->GetAsConstType(), numElements);
}


const ArrayType *
ArrayType::GetAsNonConstType() const {
    return new ArrayType(child->GetAsNonConstType(), numElements);
}


int
ArrayType::GetElementCount() const {
    return numElements;
}


const Type *
ArrayType::GetElementType() const {
    return child;
}


std::string
ArrayType::GetString() const {
    std::string s = GetBaseType()->GetString();

    const ArrayType *at = this;
    // Walk through this and any children arrays and print all of their
    // dimensions
    while (at) {
        char buf[16];
        if (numElements > 0)
            sprintf(buf, "%d", at->numElements);
        else
            buf[0] = '\0';
        s += std::string("[") + std::string(buf) + std::string("]");
        at = dynamic_cast<const ArrayType *>(at->child);
    }
    return s;
}


std::string
ArrayType::Mangle() const {
    std::string s = child->Mangle();
    char buf[16];
    if (numElements > 0)
        sprintf(buf, "%d", numElements);
    else
        buf[0] = '\0';
    return s + "[" + buf + "]";
}


std::string
ArrayType::GetCDeclaration(const std::string &name) const {
    std::string s = GetBaseType()->GetCDeclaration(name);

    const ArrayType *at = this;
    while (at) {
        char buf[16];
        if (numElements > 0)
            sprintf(buf, "%d", at->numElements);
        else
            buf[0] = '\0';
        s += std::string("[") + std::string(buf) + std::string("]");
        at = dynamic_cast<const ArrayType *>(at->child);
    }
    return s;
}


int
ArrayType::TotalElementCount() const {
    const ArrayType *ct = dynamic_cast<const ArrayType *>(child);
    if (ct)
        return numElements * ct->TotalElementCount();
    else
        return numElements;
}


llvm::DIType
ArrayType::GetDIType(llvm::DIDescriptor scope) const {
#ifdef LLVM_2_8
    FATAL("debug info not supported in llvm 2.8");
    return llvm::DIType();
#else
    if (!child)
        return llvm::DIType();

    llvm::DIType eltType = child->GetDIType(scope);

    int lowerBound = 0, upperBound = numElements-1;
    if (numElements == 0) {
        // unsized array -> indicate with low > high
        lowerBound = 1;
        upperBound = 0;
    }

    llvm::Value *sub = m->diBuilder->getOrCreateSubrange(lowerBound, upperBound);
    std::vector<llvm::Value *> subs;
    subs.push_back(sub);
#ifdef LLVM_2_9
    llvm::DIArray subArray = m->diBuilder->getOrCreateArray(&subs[0], subs.size());
#else
    llvm::DIArray subArray = m->diBuilder->getOrCreateArray(subs);
#endif

    // it's intentional that size is zero for unsized arrays
    uint64_t size = eltType.getSizeInBits() * numElements;
    uint64_t align = eltType.getAlignInBits();

    return m->diBuilder->createArrayType(size, align, eltType, subArray);
#endif // LLVM_2_8
}


ArrayType *
ArrayType::GetSizedArray(int sz) const {
    assert(numElements == 0);
    return new ArrayType(child, sz);
}


///////////////////////////////////////////////////////////////////////////
// SOAArrayType

SOAArrayType::SOAArrayType(const StructType *eltType, int nElem, int sw) 
    : ArrayType(eltType, nElem), soaWidth(sw) {
    assert(soaWidth > 0);
    if (numElements > 0)
        assert((numElements % soaWidth) == 0);
}


// FIXME: do we need to implement GetBaseType() here to return child->SOAType()?

const SOAArrayType *
SOAArrayType::GetAsVaryingType() const {
    return new SOAArrayType(dynamic_cast<const StructType *>(child->GetAsVaryingType()), 
                            numElements, soaWidth);
}


const SOAArrayType *
SOAArrayType::GetAsUniformType() const {
    return new SOAArrayType(dynamic_cast<const StructType *>(child->GetAsUniformType()), 
                            numElements, soaWidth);
}


const Type *
SOAArrayType::GetSOAType(int width) const {
    return new SOAArrayType(dynamic_cast<const StructType *>(child->GetSOAType(width)), 
                            numElements, soaWidth);
}


const SOAArrayType *
SOAArrayType::GetAsConstType() const {
    return new SOAArrayType(dynamic_cast<const StructType *>(child->GetAsConstType()), 
                            numElements, soaWidth);
}


const SOAArrayType *
SOAArrayType::GetAsNonConstType() const {
    return new SOAArrayType(dynamic_cast<const StructType *>(child->GetAsNonConstType()), 
                            numElements, soaWidth);
}


std::string
SOAArrayType::GetString() const {
    std::string s;

    char buf[32];
    sprintf(buf, "soa<%d> ", soaWidth);
    s += buf;
    s += GetBaseType()->GetString();

    const ArrayType *at = this;
    while (at) {
        char buf[16];
        if (numElements > 0)
            sprintf(buf, "%d", at->numElements);
        else
            buf[0] = '\0';
        s += std::string("[") + std::string(buf) + std::string("]");
        at = dynamic_cast<const ArrayType *>(at->child);
    }
    return s;
}


std::string
SOAArrayType::Mangle() const {
    const Type *t = soaType();
    return t->Mangle();
}


std::string
SOAArrayType::GetCDeclaration(const std::string &name) const {
    const Type *t = soaType();
    return t->GetCDeclaration(name);
}


int
SOAArrayType::TotalElementCount() const {
    int sz = numElements / soaWidth;
    const ArrayType *ct = dynamic_cast<const ArrayType *>(child);
    if (ct)
        return sz * ct->TotalElementCount();
    else
        return sz;
}


LLVM_TYPE_CONST llvm::ArrayType *
SOAArrayType::LLVMType(llvm::LLVMContext *ctx) const {
    if (!child)
        return NULL;

    const ArrayType *a = soaType();
    if (!a)
        return NULL;
    return a->LLVMType(ctx);
}


llvm::DIType
SOAArrayType::GetDIType(llvm::DIDescriptor scope) const {
#ifdef LLVM_2_8
    FATAL("debug info not supported in llvm 2.8");
    return llvm::DIType();
#else
    if (!child)
        return llvm::DIType();

    const Type *t = soaType();
    return t->GetDIType(scope);
#endif
}


SOAArrayType *
SOAArrayType::GetSizedArray(int size) const {
    if ((size % soaWidth) != 0)
        return NULL;
    return new SOAArrayType(dynamic_cast<const StructType *>(child), size, soaWidth);
}


const ArrayType *
SOAArrayType::soaType() const {
    const Type *childSOA = child->GetSOAType(soaWidth);
    return new ArrayType(childSOA, numElements / soaWidth);
}


///////////////////////////////////////////////////////////////////////////
// VectorType

VectorType::VectorType(const AtomicType *b, int a) 
    : base(b), numElements(a) {
    assert(numElements > 0);
    assert(base != NULL);
}


bool
VectorType::IsUniformType() const {
    return base->IsUniformType(); 
}


bool
VectorType::IsFloatType() const {
    return base->IsFloatType(); 
}


bool
VectorType::IsIntType() const {
    return base->IsIntType(); 
}


bool
VectorType::IsUnsignedType() const {
    return base->IsUnsignedType(); 
}


bool
VectorType::IsBoolType() const {
    return base->IsBoolType(); 
}


bool
VectorType::IsConstType() const {
    return base->IsConstType(); 
}


const Type *
VectorType::GetBaseType() const {
    return base;
}


const VectorType *
VectorType::GetAsVaryingType() const {
    return new VectorType(base->GetAsVaryingType(), numElements);
}


const VectorType *
VectorType::GetAsUniformType() const {
    return new VectorType(base->GetAsUniformType(), numElements);
}


const Type *
VectorType::GetSOAType(int width) const {
    // FIXME: is this right??
    return new ArrayType(this, width);
}


const VectorType *
VectorType::GetAsConstType() const {
    return new VectorType(base->GetAsConstType(), numElements);
}


const VectorType *
VectorType::GetAsNonConstType() const {
    return new VectorType(base->GetAsNonConstType(), numElements);
}


std::string
VectorType::GetString() const {
    std::string s = base->GetString();
    char buf[16];
    sprintf(buf, "<%d>", numElements);
    return s + std::string(buf);
}


std::string
VectorType::Mangle() const {
    std::string s = base->Mangle();
    char buf[16];
    sprintf(buf, "<%d>", numElements);
    return s + std::string(buf);
}


std::string
VectorType::GetCDeclaration(const std::string &name) const {
    std::string s = base->GetCDeclaration("");
    char buf[16];
    sprintf(buf, "%d", numElements);
    return s + std::string(buf) + "  " + name;
}


int
VectorType::GetElementCount() const {
    return numElements;
}


const AtomicType *
VectorType::GetElementType() const {
    return base;
}


LLVM_TYPE_CONST llvm::Type *
VectorType::LLVMType(llvm::LLVMContext *ctx) const {
    LLVM_TYPE_CONST llvm::Type *bt = base->LLVMType(ctx);
    if (!bt)
        return NULL;

    if (base->IsUniformType())
        // vectors of uniform types are laid out across LLVM vectors, with
        // the llvm vector size set to be a multiple of the machine's
        // natural vector size (e.g. 4 on SSE).  This is a roundabout way
        // of ensuring that LLVM lays them out into machine vector
        // registers so that e.g. if we want to add two uniform 4 float
        // vectors, that is turned into a single addps on SSE.
        return llvm::VectorType::get(bt, getVectorMemoryCount());
    else
        // varying types are already laid out to fill HW vector registers,
        // so a vector type here is just expanded out as an llvm array.
        return llvm::ArrayType::get(bt, getVectorMemoryCount());
}


llvm::DIType
VectorType::GetDIType(llvm::DIDescriptor scope) const {
#ifdef LLVM_2_8
    FATAL("debug info not supported in llvm 2.8");
    return llvm::DIType();
#else
    llvm::DIType eltType = base->GetDIType(scope);
    llvm::Value *sub = m->diBuilder->getOrCreateSubrange(0, numElements-1);
#ifdef LLVM_2_9
    llvm::Value *subs[1] = { sub };
    llvm::DIArray subArray = m->diBuilder->getOrCreateArray(subs, 1);
#else
    llvm::DIArray subArray = m->diBuilder->getOrCreateArray(sub);
#endif

    uint64_t sizeBits = eltType.getSizeInBits() * numElements;

    // vectors of varying types are already naturally aligned to the
    // machine's vector width, but arrays of uniform types need to be
    // explicitly aligned to the machines natural vector alignment.
    uint64_t align = eltType.getAlignInBits();
    if (IsUniformType())
        align = 4 * g->target.nativeVectorWidth;

    return m->diBuilder->createVectorType(sizeBits, align, eltType, subArray);
#endif // LLVM_2_8
}


int
VectorType::getVectorMemoryCount() const {
    if (base->IsVaryingType())
        return numElements;
    else {
        int nativeWidth = g->target.nativeVectorWidth;
        if (base->GetAsUniformType() == AtomicType::UniformInt64 ||
            base->GetAsUniformType() == AtomicType::UniformUInt64 ||
            base->GetAsUniformType() == AtomicType::UniformDouble)
            // target.nativeVectorWidth should be in terms of 32-bit
            // values, so for the 64-bit guys, it takes half as many of
            // them to fill the native width
            nativeWidth /= 2;
        // and now round up the element count to be a multiple of
        // nativeWidth
        return (numElements + (nativeWidth - 1)) & ~(nativeWidth-1);
    }
}


///////////////////////////////////////////////////////////////////////////
// StructType

StructType::StructType(const std::string &n, const std::vector<const Type *> &elts, 
                       const std::vector<std::string> &en,
                       const std::vector<SourcePos> &ep,
                       bool ic, bool iu, SourcePos p) 
    : name(n), elementTypes(elts), elementNames(en), elementPositions(ep),
      isUniform(iu), isConst(ic), pos(p) {
}


bool
StructType::IsUniformType() const  {
    return isUniform; 
}


bool
StructType::IsBoolType() const {
    return false; 
}


bool
StructType::IsFloatType() const {
    return false; 
}


bool
StructType::IsIntType() const  {
    return false; 
}


bool
StructType::IsUnsignedType() const {
    return false; 
}


bool
StructType::IsConstType() const {
    return isConst; 
}


const Type *
StructType::GetBaseType() const {
    return this;
}


const StructType *
StructType::GetAsVaryingType() const {
    if (IsVaryingType()) 
        return this;
    else
        return new StructType(name, elementTypes, elementNames, elementPositions,
                              isConst, false, pos);
}


const StructType *
StructType::GetAsUniformType() const {
    if (IsUniformType()) 
        return this;
    else
        return new StructType(name, elementTypes, elementNames, elementPositions,
                              isConst, true, pos);
}


const Type *
StructType::GetSOAType(int width) const {
    std::vector<const Type *> et;
    // The SOA version of a structure is just a structure that holds SOAed
    // versions of its elements
    for (int i = 0; i < GetElementCount(); ++i) {
        const Type *t = GetElementType(i);
        et.push_back(t->GetSOAType(width));
    }
    return new StructType(name, et, elementNames, elementPositions,
                          isConst, isUniform, pos);
}


const StructType *
StructType::GetAsConstType() const {
    if (IsConstType()) 
        return this;
    else
        return new StructType(name, elementTypes, elementNames, 
                              elementPositions, true, isUniform, pos);
}


const StructType *
StructType::GetAsNonConstType() const {
    if (!IsConstType()) 
        return this;
    else
        return new StructType(name, elementTypes, elementNames, elementPositions,
                              false, isUniform, pos);
}


std::string
StructType::GetString() const {
    std::string ret;
    if (isConst)   ret += "const ";
    if (isUniform) ret += "uniform ";
    else           ret += "varying ";

    // Don't print the entire struct declaration, just print the struct's name.
    // @todo Do we need a separate method that prints the declaration?
#if 0
    ret += std::string("struct { ") + name;
    for (unsigned int i = 0; i < elementTypes.size(); ++i) {
        ret += elementTypes[i]->GetString();
        ret += " ";
        ret += elementNames[i];
        ret += "; ";
    }
    ret += "}";
#else
    ret += "struct ";
    ret += name;
#endif
    return ret;
}


std::string
StructType::Mangle() const {
    std::string ret;
    ret += "s[";
    if (isConst)
        ret += "_c_";
    if (isUniform)
        ret += "_u_";
    ret += name + std::string("]<");
    for (unsigned int i = 0; i < elementTypes.size(); ++i)
        ret += elementTypes[i]->Mangle();
    ret += ">";
    return ret;
}
    

std::string
StructType::GetCDeclaration(const std::string &n) const {
    std::string ret;
    if (isConst) ret += "const ";
    ret += std::string("struct ") + name;
    if (lShouldPrintName(n))
        ret += std::string(" ") + n;
    if (!isUniform) {
        char buf[16];
        sprintf(buf, "[%d]", g->target.vectorWidth);
        ret += buf;
    }
    return ret;
}


LLVM_TYPE_CONST llvm::Type *
StructType::LLVMType(llvm::LLVMContext *ctx) const {
    std::vector<LLVM_TYPE_CONST llvm::Type *> llvmTypes;
    for (int i = 0; i < GetElementCount(); ++i) {
        const Type *type = GetElementType(i);
        llvmTypes.push_back(type->LLVMType(ctx));
    }
    return llvm::StructType::get(*ctx, llvmTypes);
}


llvm::DIType
StructType::GetDIType(llvm::DIDescriptor scope) const {
#ifdef LLVM_2_8
    FATAL("debug info not supported in llvm 2.8");
    return llvm::DIType();
#else
    uint64_t currentSize = 0, align = 0;

    std::vector<llvm::Value *> elementLLVMTypes;
    // Walk through the elements of the struct; for each one figure out its
    // alignment and size, using that to figure out its offset w.r.t. the
    // start of the structure.
    for (unsigned int i = 0; i < elementTypes.size(); ++i) {
        llvm::DIType eltType = GetElementType(i)->GetDIType(scope);
        uint64_t eltAlign = eltType.getAlignInBits();
        uint64_t eltSize = eltType.getSizeInBits();

        // The alignment for the entire structure is the maximum of the
        // required alignments of its elements
        align = std::max(align, eltAlign);

        // Move the current size forward if needed so that the current
        // element starts at an offset that's the correct alignment.
        if (currentSize > 0 && (currentSize % eltAlign))
            currentSize += eltAlign - (currentSize % eltAlign);
        assert((currentSize == 0) || (currentSize % eltAlign) == 0);

        llvm::DIFile diFile = elementPositions[i].GetDIFile();
        int line = elementPositions[i].first_line;
#ifdef LLVM_2_9
        llvm::DIType fieldType = 
            m->diBuilder->createMemberType(elementNames[i], diFile, line,
                                           eltSize, eltAlign, currentSize, 0,
                                           eltType);
#else
        llvm::DIType fieldType = 
            m->diBuilder->createMemberType(scope, elementNames[i], diFile, 
                                           line, eltSize, eltAlign, 
                                           currentSize, 0, eltType);
#endif // LLVM_2_9
        elementLLVMTypes.push_back(fieldType);

        currentSize += eltSize;
    }

    // Round up the struct's entire size so that it's a multiple of the
    // required alignment that we figured out along the way...
    if (currentSize > 0 && (currentSize % align))
        currentSize += align - (currentSize % align);

#ifdef LLVM_2_9
    llvm::DIArray elements = m->diBuilder->getOrCreateArray(&elementLLVMTypes[0], 
                                                            elementLLVMTypes.size());
#else
    llvm::DIArray elements = m->diBuilder->getOrCreateArray(elementLLVMTypes);
#endif
    llvm::DIFile diFile = pos.GetDIFile();
    return m->diBuilder->createStructType(scope, name, diFile, pos.first_line, currentSize, 
                                          align, 0, elements);
#endif // LLVM_2_8
}


const Type *
StructType::GetElementType(int i) const {
    assert(i < (int)elementTypes.size());
    // If the struct is uniform qualified, then each member comes out with
    // the same type as in the original source file.  If it's varying, then
    // all members are promoted to varying.
    const Type *ret = isUniform ? elementTypes[i] : 
        elementTypes[i]->GetAsVaryingType();
    return isConst ? ret->GetAsConstType() : ret;
}


const Type *
StructType::GetElementType(const std::string &n) const {
    for (unsigned int i = 0; i < elementNames.size(); ++i)
        if (elementNames[i] == n) {
            const Type *ret = isUniform ? elementTypes[i] : 
                elementTypes[i]->GetAsVaryingType();
            return isConst ? ret->GetAsConstType() : ret;
        }
    return NULL;
}


int
StructType::GetElementNumber(const std::string &n) const {
    for (unsigned int i = 0; i < elementNames.size(); ++i)
        if (elementNames[i] == n)
            return i;
    return -1;
}


///////////////////////////////////////////////////////////////////////////
// ReferenceType

ReferenceType::ReferenceType(const Type *t, bool ic) 
    : isConst(ic), targetType(t->GetAsNonConstType()) {
}


bool
ReferenceType::IsUniformType() const {
    return targetType->IsUniformType(); 
}


bool
ReferenceType::IsBoolType() const {
    return targetType->IsBoolType(); 
}


bool
ReferenceType::IsFloatType() const {
    return targetType->IsFloatType(); 
}


bool
ReferenceType::IsIntType() const {
    return targetType->IsIntType(); 
}


bool
ReferenceType::IsUnsignedType() const {
    return targetType->IsUnsignedType(); 
}


bool
ReferenceType::IsConstType() const {
    return isConst; 
}


const Type *
ReferenceType::GetReferenceTarget() const {
    return targetType;
}


const Type *
ReferenceType::GetBaseType() const {
    return targetType->GetBaseType();
}


const ReferenceType *
ReferenceType::GetAsVaryingType() const {
    if (IsVaryingType()) 
        return this;
    return new ReferenceType(targetType->GetAsVaryingType(), isConst);
}


const ReferenceType *
ReferenceType::GetAsUniformType() const {
    if (IsUniformType()) 
        return this;
    return new ReferenceType(targetType->GetAsUniformType(), isConst);
}


const Type *
ReferenceType::GetSOAType(int width) const {
    return new ReferenceType(targetType->GetSOAType(width), isConst);
}


const ReferenceType *
ReferenceType::GetAsConstType() const {
    if (IsConstType())
        return this;
    return new ReferenceType(targetType, true);
}


const ReferenceType *
ReferenceType::GetAsNonConstType() const {
    if (!IsConstType())
        return this;
    return new ReferenceType(targetType, false);
}


std::string
ReferenceType::GetString() const {
    std::string ret;
    if (isConst || targetType->IsConstType())
        ret += "const ";
    ret += std::string("reference<") + targetType->GetAsNonConstType()->GetString() + 
        std::string(">");
    return ret;
}


std::string
ReferenceType::Mangle() const {
    std::string ret;
    if (isConst)
        ret += "C";
    ret += std::string("REF") + targetType->Mangle();
    return ret;
}


std::string
ReferenceType::GetCDeclaration(const std::string &name) const {
    const ArrayType *at = dynamic_cast<const ArrayType *>(targetType);
    if (at != NULL) {
        if (at->GetElementCount() == 0) {
            // emit unsized arrays as pointers to the base type..
            std::string ret;
            if (isConst || at->GetElementType()->IsConstType())
                ret += "const ";
            ret += at->GetElementType()->GetAsNonConstType()->GetCDeclaration("") + 
                std::string(" *");
            if (lShouldPrintName(name))
                ret += name;
            return ret;
        }
        else
            // otherwise forget about the reference part if it's an
            // array since C already passes arrays by reference...
            return targetType->GetCDeclaration(name);
    }
    else {
        std::string ret;
        if (isConst || targetType->IsConstType())
            ret += "const ";
        ret += targetType->GetAsNonConstType()->GetCDeclaration("") + 
            std::string(" *");
        if (lShouldPrintName(name))
            ret += name;
        return ret;
    }
}


LLVM_TYPE_CONST llvm::Type *
ReferenceType::LLVMType(llvm::LLVMContext *ctx) const {
    if (!targetType)
        return NULL;
    LLVM_TYPE_CONST llvm::Type *t = targetType->LLVMType(ctx);
    if (!t)
        return NULL;
    return llvm::PointerType::get(t, 0);
}


llvm::DIType
ReferenceType::GetDIType(llvm::DIDescriptor scope) const {
#ifdef LLVM_2_8
    FATAL("debug info not supported in llvm 2.8");
    return llvm::DIType();
#else
    llvm::DIType diTargetType = targetType->GetDIType(scope);
    return m->diBuilder->createReferenceType(diTargetType);
#endif // LLVM_2_8
}


///////////////////////////////////////////////////////////////////////////
// FunctionType

FunctionType::FunctionType(const Type *r, const std::vector<const Type *> &a, 
                           SourcePos p, const std::vector<std::string> *an, 
                           bool it, bool is, bool ec) 
    : isTask(it), isExported(is), isExternC(ec), returnType(r), argTypes(a), 
      argNames(an ? *an : std::vector<std::string>()), pos(p) {
    assert(returnType != NULL);
}


bool
FunctionType::IsUniformType() const {
    return returnType->IsUniformType(); 
}


bool
FunctionType::IsFloatType() const {
    return returnType->IsFloatType(); 
}


bool
FunctionType::IsIntType() const {
    return returnType->IsIntType(); 
}


bool
FunctionType::IsBoolType() const {
    return returnType->IsBoolType(); 
}


bool
FunctionType::IsUnsignedType() const {
    return returnType->IsUnsignedType(); 
}


bool
FunctionType::IsConstType() const {
    return returnType->IsConstType(); 
}


const Type *
FunctionType::GetBaseType() const {
    FATAL("FunctionType::GetBaseType() shouldn't be called");
    return NULL;
}


const Type *
FunctionType::GetAsVaryingType() const {
    FATAL("FunctionType::GetAsVaryingType shouldn't be called");
    return NULL;
}


const Type *
FunctionType::GetAsUniformType() const {
    FATAL("FunctionType::GetAsUniformType shouldn't be called");
    return NULL;
}


const Type *
FunctionType::GetSOAType(int width) const {
    FATAL("FunctionType::GetSOAType shouldn't be called");
    return NULL;
}


const Type *
FunctionType::GetAsConstType() const {
    FATAL("FunctionType::GetAsConstType shouldn't be called");
    return NULL;
}


const Type *
FunctionType::GetAsNonConstType() const {
    FATAL("FunctionType::GetAsNonConstType shouldn't be called");
    return NULL;
}


std::string
FunctionType::GetString() const {
    std::string ret;
    if (isTask) ret += "task ";
    ret += returnType->GetString();
    ret += "(";
    for (unsigned int i = 0; i < argTypes.size(); ++i) {
        ret += argTypes[i]->GetString();
        if (i != argTypes.size() - 1)
            ret += ", ";
    }
    ret += ")";
    return ret;
}


std::string
FunctionType::Mangle() const {
    std::string ret = "___";
    for (unsigned int i = 0; i < argTypes.size(); ++i)
        ret += argTypes[i]->Mangle();
    return ret;
}


std::string
FunctionType::GetCDeclaration(const std::string &fname) const {
    std::string ret;
    ret += returnType->GetCDeclaration("");
    ret += " ";
    ret += fname;
    ret += "(";
    for (unsigned int i = 0; i < argTypes.size(); ++i) {
        if (argNames.size())
            ret += argTypes[i]->GetCDeclaration(argNames[i]);
        else
            ret += argTypes[i]->GetString();
        if (i != argTypes.size() - 1)
            ret += ", ";
    }
    ret += ")";
    return ret;
}


LLVM_TYPE_CONST llvm::Type *
FunctionType::LLVMType(llvm::LLVMContext *ctx) const {
    FATAL("FunctionType::LLVMType() shouldn't be called");
    return NULL;
}


llvm::DIType
FunctionType::GetDIType(llvm::DIDescriptor scope) const {
    // @todo need to implement FunctionType::GetDIType()
    FATAL("need to implement FunctionType::GetDIType()");
    return llvm::DIType();
}


LLVM_TYPE_CONST llvm::FunctionType *
FunctionType::LLVMFunctionType(llvm::LLVMContext *ctx, bool includeMask) const {
    if (!includeMask && isTask) {
        Error(pos, "Function can't have both \"task\" and \"export\" qualifiers");
        return NULL;
    }

    // Get the LLVM Type *s for the function arguments
    std::vector<LLVM_TYPE_CONST llvm::Type *> llvmArgTypes;
    for (unsigned int i = 0; i < argTypes.size(); ++i) {
        if (!argTypes[i])
            return NULL;

        LLVM_TYPE_CONST llvm::Type *t = argTypes[i]->LLVMType(ctx);
        if (!t)
            return NULL;
        llvmArgTypes.push_back(t);
    }

    // And add the function mask, if asked for
    if (includeMask)
        llvmArgTypes.push_back(LLVMTypes::MaskType);

    std::vector<LLVM_TYPE_CONST llvm::Type *> callTypes;
    if (isTask) {
        // Tasks take three arguments: a pointer to a struct that holds the
        // actual task arguments, the thread index, and the total number of
        // threads the tasks system has running.  (Task arguments are
        // marshalled in a struct so that it's easy to allocate space to
        // hold them until the task actually runs.)
        llvm::Type *st = llvm::StructType::get(*ctx, llvmArgTypes);
        callTypes.push_back(llvm::PointerType::getUnqual(st));
        callTypes.push_back(LLVMTypes::Int32Type); // threadIndex
        callTypes.push_back(LLVMTypes::Int32Type); // threadCount
    }
    else
        // Otherwise we already have the types of the arguments 
        callTypes = llvmArgTypes;

    return llvm::FunctionType::get(returnType->LLVMType(g->ctx), callTypes, false);
}


void
FunctionType::SetArgumentDefaults(const std::vector<ConstExpr *> &d) const {
    assert(argDefaults.size() == 0);
    assert(d.size() == argTypes.size());
    argDefaults = d;
}


///////////////////////////////////////////////////////////////////////////
// Type

const Type *
Type::GetReferenceTarget() const {
    // only ReferenceType needs to override this method
    return this;
}


const Type *
Type::GetAsUnsignedType() const {
    // For many types, this doesn't make any sesne
    return NULL;
}


/** Given an atomic or vector type, return a vector type of the given
    vecSize.  Issue an error if given a vector type that isn't already that
    size.
 */
static const Type *
lVectorConvert(const Type *type, SourcePos pos, const char *reason, int vecSize) {
    const VectorType *vt = dynamic_cast<const VectorType *>(type);
    if (vt) {
        if (vt->GetElementCount() != vecSize) {
            Error(pos, "Implicit conversion between from vector type "
                  "\"%s\" to vector type of length %d for %s is not possible.", 
                  type->GetString().c_str(), vecSize, reason);
            return NULL;
        }
        return vt;
    }
    else {
        const AtomicType *at = dynamic_cast<const AtomicType *>(type);
        if (!at) {
            Error(pos, "Non-atomic type \"%s\" can't be converted to vector type "
                  "for %s.", type->GetString().c_str(), reason);
            return NULL;
        }
        return new VectorType(at, vecSize);
    }
}


const Type *
Type::MoreGeneralType(const Type *t0, const Type *t1, SourcePos pos, const char *reason, 
                      bool forceVarying, int vecSize) {
    assert(reason != NULL);

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

    // Not the same types, but only a const/non-const difference?  Return
    // the non-const type as the more general one.
    if (Type::Equal(t0->GetAsConstType(), t1->GetAsConstType()))
        return t0->GetAsNonConstType();

    const VectorType *vt0 = dynamic_cast<const VectorType *>(t0);
    const VectorType *vt1 = dynamic_cast<const VectorType *>(t1);
    if (vt0 && vt1) {
        // both are vectors; convert their base types and make a new vector
        // type, as long as their lengths match
        if (vt0->GetElementCount() != vt1->GetElementCount()) {
            Error(pos, "Implicit conversion between differently sized vector types "
                  "(%s, %s) for %s is not possible.", t0->GetString().c_str(),
                  t1->GetString().c_str(), reason);
            return NULL;
        }
        const Type *t = MoreGeneralType(vt0->GetElementType(), vt1->GetElementType(),
                                        pos, reason, forceVarying);
        if (!t) 
            return NULL;

        // The 'more general' version of the two vector element types must
        // be an AtomicType (that's all that vectors can hold...)
        const AtomicType *at = dynamic_cast<const AtomicType *>(t);
        assert(at != NULL);

        return new VectorType(at, vt0->GetElementCount());
    }
    else if (vt0) {
        // If one type is a vector type but the other isn't, see if we can
        // promote the other one to a vector type.  This will fail and
        // return NULL if t1 is e.g. an array type and it's illegal to have
        // a vector of it..
        const Type *t = MoreGeneralType(vt0->GetElementType(), t1, pos, 
                                        reason, forceVarying);
        if (!t) 
            return NULL;

        const AtomicType *at = dynamic_cast<const AtomicType *>(t);
        assert(at != NULL);
        return new VectorType(at, vt0->GetElementCount());
    }
    else if (vt1) {
        // As in the above case, see if we can promote t0 to make a vector
        // that matches vt1.
        const Type *t = MoreGeneralType(t0, vt1->GetElementType(), pos, 
                                        reason, forceVarying);
        if (!t) 
            return NULL;

        const AtomicType *at = dynamic_cast<const AtomicType *>(t);
        assert(at != NULL);
        return new VectorType(at, vt1->GetElementCount());
    }

    // TODO: what do we need to do about references here, if anything??

    const AtomicType *at0 = dynamic_cast<const AtomicType *>(t0->GetReferenceTarget());
    const AtomicType *at1 = dynamic_cast<const AtomicType *>(t1->GetReferenceTarget());

    const EnumType *et0 = dynamic_cast<const EnumType *>(t0->GetReferenceTarget());
    const EnumType *et1 = dynamic_cast<const EnumType *>(t1->GetReferenceTarget());
    if (et0 != NULL && et1 != NULL) {
        // Two different enum types -> make them uint32s...
        assert(et0->IsVaryingType() == et1->IsVaryingType());
        return et0->IsVaryingType() ? AtomicType::VaryingUInt32 :
                AtomicType::UniformUInt32;
    }
    else if (et0 != NULL) {
        if (at1 != NULL)
            // Enum type and atomic type -> convert the enum to the atomic type
            // TODO: should we return uint32 here, unless the atomic type is
            // a 64-bit atomic type, in which case we return that?
            return at1;
        else {
            Error(pos, "Implicit conversion from enum type \"%s\" to "
                  "non-atomic type \"%s\" for %s not possible.",
                  t0->GetString().c_str(), t1->GetString().c_str(), reason);
            return NULL;
        }
    }
    else if (et1 != NULL) {
        if (at0 != NULL)
            // Enum type and atomic type; see TODO above here as well...
            return at0;
        else {
            Error(pos, "Implicit conversion from enum type \"%s\" to "
                  "non-atomic type \"%s\" for %s not possible.",
                  t1->GetString().c_str(), t0->GetString().c_str(), reason);
            return NULL;
        }
    }

    // Now all we can do is promote atomic types...
    if (at0 == NULL || at1 == NULL) {
        assert(reason != NULL);
        Error(pos, "Implicit conversion from type \"%s\" to \"%s\" for %s not possible.",
              t0->GetString().c_str(), t1->GetString().c_str(), reason);
        return NULL;
    }

    // Finally, to determine which of the two atomic types is more general,
    // use the ordering of entries in the AtomicType::BasicType enumerator.
    return (int(at0->basicType) >= int(at1->basicType)) ? at0 : at1;
}


bool
Type::Equal(const Type *a, const Type *b) {
    if (a == NULL || b == NULL)
        return false;

    // We can compare AtomicTypes with pointer equality, since the
    // AtomicType constructor is private so that there isonly the single
    // canonical instance of the AtomicTypes (AtomicType::UniformInt32,
    // etc.)
    if (dynamic_cast<const AtomicType *>(a) != NULL &&
        dynamic_cast<const AtomicType *>(b) != NULL)
        return a == b;

    // For all of the other types, we need to see if we have the same two
    // general types.  If so, then we dig into the details of the type and
    // see if all of the relevant bits are equal...
    const EnumType *eta = dynamic_cast<const EnumType *>(a);
    const EnumType *etb = dynamic_cast<const EnumType *>(b);
    if (eta != NULL && etb != NULL)
        // Kind of goofy, but this sufficies to check
        return (eta->pos == etb->pos &&
                eta->IsUniformType() == etb->IsUniformType() &&
                eta->IsConstType() == etb->IsConstType());

    const ArrayType *ata = dynamic_cast<const ArrayType *>(a);
    const ArrayType *atb = dynamic_cast<const ArrayType *>(b);
    if (ata != NULL && atb != NULL)
        return (ata->GetElementCount() == atb->GetElementCount() && 
                Equal(ata->GetElementType(), atb->GetElementType()));

    const VectorType *vta = dynamic_cast<const VectorType *>(a);
    const VectorType *vtb = dynamic_cast<const VectorType *>(b);
    if (vta != NULL && vtb != NULL)
        return (vta->GetElementCount() == vtb->GetElementCount() && 
                Equal(vta->GetElementType(), vtb->GetElementType()));

    const StructType *sta = dynamic_cast<const StructType *>(a);
    const StructType *stb = dynamic_cast<const StructType *>(b);
    if (sta != NULL && stb != NULL) {
        if (sta->GetElementCount() != stb->GetElementCount())
            return false;
        for (int i = 0; i < sta->GetElementCount(); ++i)
            if (!Equal(sta->GetElementType(i), stb->GetElementType(i)))
                return false;
        return true;
    }

    const ReferenceType *rta = dynamic_cast<const ReferenceType *>(a);
    const ReferenceType *rtb = dynamic_cast<const ReferenceType *>(b);
    if (rta != NULL && rtb != NULL)
        return ((rta->IsConstType() == rtb->IsConstType()) &&
                Type::Equal(rta->GetReferenceTarget(),
                            rtb->GetReferenceTarget()));

    const FunctionType *fta = dynamic_cast<const FunctionType *>(a);
    const FunctionType *ftb = dynamic_cast<const FunctionType *>(b);
    if (fta != NULL && ftb != NULL) {
        // Both the return types and all of the argument types must match
        // for function types to match
        if (!Equal(fta->GetReturnType(), ftb->GetReturnType()))
            return false;

        const std::vector<const Type *> &aargs = fta->GetArgumentTypes();
        const std::vector<const Type *> &bargs = ftb->GetArgumentTypes();
        if (aargs.size() != bargs.size())
            return false;
        for (unsigned int i = 0; i < aargs.size(); ++i)
            if (!Equal(aargs[i], bargs[i]))
                return false;
        return true;
    }

    return false;
}
