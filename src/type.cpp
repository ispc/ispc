/*
  Copyright (c) 2010-2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file type.cpp
    @brief Definitions for classes related to type representation
*/

#include "type.h"
#include "expr.h"
#include "func.h"
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
    if (name.size() == 0) {
        return false;
    } else if (name[0] != '_' && name[0] != '$') {
        return true;
    } else {
        return (name.size() == 1) || (name[1] != '_');
    }
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
        // We need to mangle unbound variability for template type arguments.
        // return "ub";
        FATAL("Unbound unexpected in Variability::MangleString()");
    default:
        FATAL("Unhandled variability");
        return "";
    }
}

///////////////////////////////////////////////////////////////////////////
// Type

Type::Type(TypeId id, Variability v, bool c, SourcePos s, unsigned int al)
    : typeId(id), variability(v), isConst(c), pos(s), alignment(al) {}

const Type *Type::createWithConst(bool newIsConst) const {
    Type *ins = create();
    ins->isConst = newIsConst;
    return ins;
}

const Type *Type::createWithVariability(Variability newVariability) const {
    Type *ins = create();
    ins->variability = newVariability;
    return ins;
}

const Type *Type::createWithAlignment(unsigned int newAlignment) const {
    const Type *ins = create();
    ins->alignment = newAlignment;
    return ins;
}

llvm::Type *Type::LLVMStorageType(llvm::LLVMContext *ctx) const { return LLVMType(ctx); }

const Type *Type::ResolveDependenceForTopType(TemplateInstantiation &templInst) const {
    const Type *temp = ResolveDependence(templInst);
    const Type *result = temp->ResolveUnboundVariability(Variability::Varying);
    return result;
}

bool Type::IsPointerType() const { return (CastType<PointerType>(this) != nullptr); }

bool Type::IsArrayType() const { return (CastType<ArrayType>(this) != nullptr); }

bool Type::IsAtomicType() const { return (CastType<AtomicType>(this) != nullptr); }

bool Type::IsVaryingAtomicOrUniformVectorType() const {
    return ((CastType<AtomicType>(this) != nullptr && IsVaryingType()) ||
            (CastType<VectorType>(this) != nullptr && IsUniformType()));
}

bool Type::IsVaryingAtomic() const { return IsAtomicType() && IsVaryingType(); }

bool Type::IsUniformVector() const { return IsVectorType() && IsUniformType(); }

bool Type::IsReferenceType() const { return (CastType<ReferenceType>(this) != nullptr); }

bool Type::IsVectorType() const { return (CastType<VectorType>(this) != nullptr); }

bool Type::IsVoidType() const { return EqualIgnoringConst(this, AtomicType::Void); }

bool Type::IsDependent() const { return IsTypeDependent() || IsCountDependent(); }

bool Type::IsTypeDependent() const {
    switch (typeId) {
    case ATOMIC_TYPE:
        return CastType<AtomicType>(this)->basicType == AtomicType::TYPE_DEPENDENT;
    case ENUM_TYPE:
        return false;
    case POINTER_TYPE: {
        const Type *baseType = CastType<PointerType>(this)->GetBaseType();
        return baseType && baseType->IsTypeDependent();
    }
    case ARRAY_TYPE: {
        const ArrayType *arrayType = CastType<ArrayType>(this);
        const Type *elemType = arrayType->GetElementType();
        return elemType && elemType->IsTypeDependent();
    }
    case VECTOR_TYPE: {
        const VectorType *vecType = CastType<VectorType>(this);
        const Type *elemType = vecType->GetElementType();
        return elemType && elemType->IsTypeDependent();
    }
    case STRUCT_TYPE: {
        const StructType *st = CastType<StructType>(this);
        for (int i = 0; i < st->GetElementCount(); ++i) {
            if (st->GetRawElementType(i)->IsTypeDependent()) {
                return true;
            }
        }
        return false;
    }
    case UNDEFINED_STRUCT_TYPE:
        return false;
    case REFERENCE_TYPE:
        return CastType<ReferenceType>(this)->GetReferenceTarget()->IsTypeDependent();
    case FUNCTION_TYPE: {
        const FunctionType *ft = CastType<FunctionType>(this);
        for (int i = 0; i < ft->GetNumParameters(); ++i) {
            if (ft->GetParameterType(i)->IsTypeDependent()) {
                return true;
            }
        }
        if (ft->GetReturnType()->IsTypeDependent()) {
            return true;
        }
        return false;
    }
    case TEMPLATE_TYPE_PARM_TYPE:
        return true;
    }
    UNREACHABLE();
}

bool Type::IsCountDependent() const {
    switch (typeId) {
    case ATOMIC_TYPE:
    case ENUM_TYPE:
    case POINTER_TYPE:
    case STRUCT_TYPE:
    case UNDEFINED_STRUCT_TYPE:
    case FUNCTION_TYPE:
    case REFERENCE_TYPE:
    case TEMPLATE_TYPE_PARM_TYPE: {
        return false;
    }
    case VECTOR_TYPE:
    case ARRAY_TYPE: {
        const SequentialType *secType = CastType<SequentialType>(this);
        return secType->IsCountDependent();
    }
    }
    UNREACHABLE();
}

const Type *Type::GetAsVaryingType() const {
    if (variability == Variability::Varying) {
        return this;
    }

    if (asVaryingType == nullptr) {
        asVaryingType = createWithVariability(Variability(Variability::Varying));
        if (variability == Variability::Uniform) {
            asVaryingType->asUniformType = this;
        }
    }
    return asVaryingType;
}

const Type *Type::ResolveUnboundVariability(Variability v) const {
    Assert(v != Variability::Unbound);
    if (variability != Variability::Unbound) {
        return this;
    }
    // In case of StructType:
    // We don't resolve the members here but leave them unbound, so that if
    // resolve to varying but later want to get the uniform version of this
    // type, for example, then we still have the information around about
    // which element types were originally unbound...
    return createWithVariability(v);
}

const Type *Type::GetAsUniformType() const {
    if (variability == Variability::Uniform) {
        return this;
    }

    if (asUniformType == nullptr) {
        asUniformType = createWithVariability(Variability(Variability::Uniform));
        if (variability == Variability::Varying) {
            asUniformType->asVaryingType = this;
        }
    }
    return asUniformType;
}

const Type *Type::GetAsUnboundVariabilityType() const {
    if (variability == Variability::Unbound) {
        return this;
    }
    // TODO!: why we cache uniform and varying types but not unbound?
    return createWithVariability(Variability(Variability::Unbound));
}

const Type *Type::GetAsSOAType(int width) const {
    if (GetSOAWidth() == width) {
        return this;
    }
    // TODO!: why we cache uniform and varying types but not SOA?
    return createWithVariability(Variability(Variability::SOA, width));
}

const Type *Type::GetReferenceTarget() const {
    // only ReferenceType needs to override this method
    return this;
}

const Type *Type::GetAsConstType() const {
    if (isConst) {
        return this;
    }

    if (asOtherConstType == nullptr) {
        asOtherConstType = createWithConst(true);
        asOtherConstType->asOtherConstType = this;
    }
    return asOtherConstType;
}

const Type *Type::GetAsNonConstType() const {
    if (!isConst) {
        return this;
    }

    if (asOtherConstType == nullptr) {
        asOtherConstType = createWithConst(false);
        asOtherConstType->asOtherConstType = this;
    }
    return asOtherConstType;
}

const Type *Type::GetAsAlignedType(unsigned int newAlignment) const {
    if (alignment == newAlignment) {
        return this;
    }

    return createWithAlignment(newAlignment);
}

const Type *Type::GetAsUnsignedType() const {
    // For many types, this doesn't make any sense
    return nullptr;
}

const Type *Type::GetAsSignedType() const {
    // For many types, this doesn't make any sense
    return nullptr;
}

///////////////////////////////////////////////////////////////////////////
// AtomicType

const AtomicType *AtomicType::UniformBool = new AtomicType(AtomicType::TYPE_BOOL, Variability::Uniform, false);
const AtomicType *AtomicType::VaryingBool = new AtomicType(AtomicType::TYPE_BOOL, Variability::Varying, false);
const AtomicType *AtomicType::VaryingInt1 = new AtomicType(AtomicType::TYPE_INT1, Variability::Varying, false);
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
const AtomicType *AtomicType::Dependent = new AtomicType(AtomicType::TYPE_DEPENDENT, Variability::Uniform, false);
const AtomicType *AtomicType::Void = new AtomicType(TYPE_VOID, Variability::Uniform, false);

AtomicType::AtomicType(BasicType bt, Variability v, bool ic) : Type(ATOMIC_TYPE, v, ic), basicType(bt) {
    asOtherConstType = nullptr;
    asUniformType = asVaryingType = nullptr;
}

AtomicType::AtomicType(const AtomicType &other)
    : Type(ATOMIC_TYPE, other.variability, other.isConst, other.pos, other.alignment), basicType(other.basicType) {}

AtomicType *AtomicType::create() const { return new AtomicType(*this); }

const AtomicType *AtomicType::createWithBasicType(BasicType newBasicType) const {
    AtomicType *ins = static_cast<AtomicType *>(create());
    ins->basicType = newBasicType;
    return ins;
}

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

bool AtomicType::IsSignedType() const {
    return (basicType == TYPE_INT1 || basicType == TYPE_INT8 || basicType == TYPE_INT16 || basicType == TYPE_INT32 ||
            basicType == TYPE_INT64);
}

bool AtomicType::IsBoolType() const { return basicType == TYPE_BOOL || basicType == TYPE_INT1; }

const AtomicType *AtomicType::GetAsUnsignedType() const {
    if (IsUnsignedType() == true) {
        return this;
    }

    if (IsIntType() == false) {
        return nullptr;
    }

    switch (basicType) {
    case TYPE_INT8:
        return createWithBasicType(TYPE_UINT8);
    case TYPE_INT16:
        return createWithBasicType(TYPE_UINT16);
    case TYPE_INT32:
        return createWithBasicType(TYPE_UINT32);
    case TYPE_INT64:
        return createWithBasicType(TYPE_UINT64);
    default:
        FATAL("Unexpected basicType in GetAsUnsignedType()");
        return nullptr;
    }
}

const AtomicType *AtomicType::GetAsSignedType() const {
    if (IsSignedType() == true) {
        return this;
    }

    if (IsIntType() == false) {
        return nullptr;
    }

    switch (basicType) {
    case TYPE_UINT8:
        return createWithBasicType(TYPE_INT8);
    case TYPE_UINT16:
        return createWithBasicType(TYPE_INT16);
    case TYPE_UINT32:
        return createWithBasicType(TYPE_INT32);
    case TYPE_UINT64:
        return createWithBasicType(TYPE_INT64);
    default:
        FATAL("Unexpected basicType in GetAsSignedType()");
        return nullptr;
    }
}

std::string AtomicType::GetString() const {
    std::string ret;
    if (isConst) {
        ret += "const ";
    }
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
    case TYPE_INT1:
        ret += "int1";
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
    case TYPE_DEPENDENT:
        ret += "<dependent type>";
        break;
    default:
        FATAL("Logic error in AtomicType::GetString()");
    }
    return ret;
}

std::string AtomicType::Mangle() const {
    Assert(basicType != TYPE_DEPENDENT);
    std::string ret;
    if (isConst) {
        ret += "C";
    }
    ret += variability.MangleString();

    switch (basicType) {
    case TYPE_VOID:
        ret += "v";
        break;
    case TYPE_BOOL:
        ret += "b";
        break;
    case TYPE_INT1:
        ret += "B";
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

std::string AtomicType::GetDeclaration(const std::string &name, DeclarationSyntax syntax) const {
    Assert(basicType != TYPE_DEPENDENT);
    std::string ret;
    if (variability == Variability::Unbound) {
        Assert(m->errorCount > 0);
        return ret;
    }
    if (isConst) {
        ret += "const ";
    }

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
        FATAL("Logic error in AtomicType::GetDeclaration()");
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
            if (isStorageType) {
                return isUniform ? LLVMTypes::BoolStorageType : LLVMTypes::BoolVectorStorageType;
            } else {
                return isUniform ? LLVMTypes::BoolType : LLVMTypes::BoolVectorType;
            }
        case AtomicType::TYPE_INT1:
            return isUniform ? LLVMTypes::Int1Type : LLVMTypes::Int1VectorType;
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
        case AtomicType::TYPE_DEPENDENT:
        default:
            FATAL("logic error in lGetAtomicLLVMType");
            return nullptr;
        }
    } else {
        ArrayType at(aType->GetAsUniformType(), variability.soaWidth);
        return at.LLVMType(ctx);
    }
}

llvm::Type *AtomicType::LLVMStorageType(llvm::LLVMContext *ctx) const {
    Assert(basicType != TYPE_DEPENDENT);
    return lGetAtomicLLVMType(ctx, this, true);
}
llvm::Type *AtomicType::LLVMType(llvm::LLVMContext *ctx) const {
    Assert(basicType != TYPE_DEPENDENT);
    return lGetAtomicLLVMType(ctx, this, false);
}

llvm::DIType *AtomicType::GetDIType(llvm::DIScope *scope) const {
    Assert(basicType != TYPE_DEPENDENT);
    Assert(variability.type != Variability::Unbound);

    if (variability.type == Variability::Uniform) {
        switch (basicType) {
        case TYPE_VOID:
            return nullptr;

        case TYPE_BOOL:
            return m->diBuilder->createBasicType("bool", 32 /* size */, llvm::dwarf::DW_ATE_unsigned);
            break;
        case TYPE_INT1:
            return m->diBuilder->createBasicType("int1", 1 /* size */, llvm::dwarf::DW_ATE_signed);
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

            return nullptr;
        }
    } else if (variability == Variability::Varying) {

        llvm::Metadata *sub = m->diBuilder->getOrCreateSubrange(0, g->target->getVectorWidth());

        llvm::DINodeArray subArray = m->diBuilder->getOrCreateArray(sub);
        llvm::DIType *unifType = GetAsUniformType()->GetDIType(scope);
        uint64_t width = g->target->getVectorWidth();
        uint64_t size = unifType->getSizeInBits() * width;
        uint64_t align = unifType->getAlignInBits() * width;
        return m->diBuilder->createVectorType(size, align, unifType, subArray);
    } else {
        Assert(variability == Variability::SOA);
        ArrayType at(GetAsUniformType(), variability.soaWidth);
        return at.GetDIType(scope);
    }
}

///////////////////////////////////////////////////////////////////////////
// TemplateTypeParmType

TemplateTypeParmType::TemplateTypeParmType(std::string n, Variability v, bool ic, SourcePos p)
    : Type(TEMPLATE_TYPE_PARM_TYPE, v, ic, p), name(n) {
    asOtherConstType = nullptr;
    asUniformType = asVaryingType = nullptr;
}

TemplateTypeParmType::TemplateTypeParmType(const TemplateTypeParmType &other)
    : Type(TEMPLATE_TYPE_PARM_TYPE, other.variability, other.isConst, other.pos, other.alignment), name(other.name) {}

TemplateTypeParmType *TemplateTypeParmType::create() const { return new TemplateTypeParmType(*this); }

bool TemplateTypeParmType::IsCompleteType() const { return false; }

// Revisit: Should soa type be supported for template type param?
const Type *TemplateTypeParmType::GetAsSOAType(int width) const {
    Error(pos, "soa type not supported for template type parameter.");
    return this;
}

const Type *TemplateTypeParmType::ResolveDependence(TemplateInstantiation &templInst) const {
    const Type *resolvedType = templInst.InstantiateType(GetName());
    if (resolvedType == nullptr) {
        // Failed to resolve the type, return
        return this;
    }

    if (variability == Variability::Unbound) {
        // Use resolved type variability
    } else if (variability == Variability::Uniform) {
        // Enforce uniform variability
        resolvedType = resolvedType->GetAsUniformType();
    } else if (variability == Variability::Varying) {
        // Enforce varying variability
        resolvedType = resolvedType->GetAsVaryingType();
    } else {
        UNREACHABLE();
    }

    if (isConst) {
        resolvedType = resolvedType->GetAsConstType();
    }
    return resolvedType;
}

std::string TemplateTypeParmType::GetName() const { return name; }

std::string TemplateTypeParmType::GetString() const {
    std::string ret;
    if (isConst) {
        ret += "const ";
    }

    ret += variability.GetString();
    ret += " ";
    ret += name;
    return ret;
}

std::string TemplateTypeParmType::Mangle() const {
    std::string ret;
    if (isConst) {
        ret += "C";
    }
    ret += variability.MangleString();
    ret += name;
    return ret;
}

std::string TemplateTypeParmType::GetDeclaration(const std::string &cname, DeclarationSyntax syntax) const {
    std::string ret;
    if (variability == Variability::Unbound) {
        Assert(m->errorCount > 0);
        return ret;
    }
    if (isConst) {
        ret += "const ";
    }
    ret += name;
    if (lShouldPrintName(cname)) {
        ret += " ";
        ret += cname;
    }
    return ret;
}

// This should never be called.
llvm::Type *TemplateTypeParmType::LLVMType(llvm::LLVMContext *ctx) const { UNREACHABLE(); }

// This should never be called.
llvm::DIType *TemplateTypeParmType::GetDIType(llvm::DIScope *scope) const { UNREACHABLE(); }

///////////////////////////////////////////////////////////////////////////
// EnumType

EnumType::EnumType(SourcePos p) : Type(ENUM_TYPE, Variability::Unbound, false, p) {
    //    name = "/* (anonymous) */";
}

EnumType::EnumType(const char *n, SourcePos p) : Type(ENUM_TYPE, Variability::Unbound, false, p), name(n) {}

EnumType::EnumType(const std::string &n, Variability v, bool ic, SourcePos p, const std::vector<Symbol *> &enums)
    : Type(ENUM_TYPE, v, ic, p), name(n), enumerators(enums) {}

EnumType::EnumType(const EnumType &other)
    : Type(ENUM_TYPE, other.variability, other.isConst, other.pos, other.alignment), name(other.name),
      enumerators(other.enumerators) {}

EnumType *EnumType::create() const { return new EnumType(*this); }

bool EnumType::IsIntType() const { return true; }

bool EnumType::IsUnsignedType() const { return true; }

std::string EnumType::GetString() const {
    std::string ret;
    if (isConst) {
        ret += "const ";
    }
    ret += variability.GetString();

    ret += " enum ";
    if (name.size()) {
        ret += name;
    }
    return ret;
}

std::string EnumType::Mangle() const {
    Assert(variability != Variability::Unbound);

    std::string ret;
    if (isConst) {
        ret += "C";
    }
    ret += variability.MangleString();
    //    ret += std::string("enum[") + name + std::string("]");
    ret += std::string("enum_5B_") + name + std::string("_5D_");
    return ret;
}

std::string EnumType::GetDeclaration(const std::string &varName, DeclarationSyntax syntax) const {
    if (variability == Variability::Unbound) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret;
    if (isConst) {
        ret += "const ";
    }
    ret += "enum";
    if (name.size()) {
        ret += std::string(" ") + name;
    }

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
        return nullptr;
    }
}

llvm::DIType *EnumType::GetDIType(llvm::DIScope *scope) const {

    std::vector<llvm::Metadata *> enumeratorDescriptors;
    for (unsigned int i = 0; i < enumerators.size(); ++i) {
        unsigned int enumeratorValue[1];
        Assert(enumerators[i]->constValue != nullptr);
        int count = enumerators[i]->constValue->GetValues(enumeratorValue);
        Assert(count == 1);

        llvm::Metadata *descriptor = m->diBuilder->createEnumerator(enumerators[i]->name, enumeratorValue[0]);
        enumeratorDescriptors.push_back(descriptor);
    }

    llvm::DINodeArray elementArray = m->diBuilder->getOrCreateArray(enumeratorDescriptors);
    llvm::DIFile *diFile = pos.GetDIFile();
    llvm::DINamespace *diSpace = pos.GetDINamespace();
    llvm::DIType *underlyingType = AtomicType::UniformInt32->GetDIType(scope);
    llvm::DIType *diType =
        m->diBuilder->createEnumerationType(diSpace, GetString(), diFile, pos.first_line, 32 /* size in bits */,
                                            32 /* align in bits */, elementArray, underlyingType,
#if ISPC_LLVM_VERSION > ISPC_LLVM_17_0
                                            0,
#endif
                                            name);
    switch (variability.type) {
    case Variability::Uniform:
        return diType;
    case Variability::Varying: {
        llvm::Metadata *sub = m->diBuilder->getOrCreateSubrange(0, g->target->getVectorWidth());

        llvm::DINodeArray subArray = m->diBuilder->getOrCreateArray(sub);
        // llvm::DebugNodeArray subArray = m->diBuilder->getOrCreateArray(sub);
        uint64_t width = g->target->getVectorWidth();
        uint64_t size = diType->getSizeInBits() * width;
        uint64_t align = diType->getAlignInBits() * width;
        return m->diBuilder->createVectorType(size, align, diType, subArray);
    }
    case Variability::SOA: {
        return lCreateDIArray(diType, variability.soaWidth);
    }
    default:
        FATAL("Unexpected variability in EnumType::GetDIType()");
        return nullptr;
    }
}

void EnumType::SetEnumerators(const std::vector<Symbol *> &e) { enumerators = e; }

int EnumType::GetEnumeratorCount() const { return (int)enumerators.size(); }

const Symbol *EnumType::GetEnumerator(int i) const { return enumerators[i]; }

///////////////////////////////////////////////////////////////////////////
// PointerType

PointerType *PointerType::Void = new PointerType(AtomicType::Void, Variability(Variability::Uniform), false);

PointerType::PointerType(const Type *t, Variability v, bool ic, unsigned int prop, AddressSpace as)
    : Type(POINTER_TYPE, v, ic), property(prop), baseType(t), addrSpace(as) {}

PointerType::PointerType(const PointerType &other)
    : Type(POINTER_TYPE, other.variability, other.isConst, other.pos, other.alignment), property(other.property),
      baseType(other.baseType), addrSpace(other.addrSpace) {}

PointerType *PointerType::create() const { return new PointerType(*this); }

const PointerType *PointerType::createWithBaseType(const Type *newBaseType) const {
    PointerType *ins = static_cast<PointerType *>(create());
    ins->baseType = newBaseType;
    return ins;
}
const PointerType *PointerType::createWithAddressSpace(AddressSpace newAddrSpace) const {
    PointerType *ins = static_cast<PointerType *>(create());
    ins->addrSpace = newAddrSpace;
    return ins;
}
const PointerType *PointerType::createWithBaseTypeAndVariability(const Type *newBaseType,
                                                                 Variability newVariability) const {
    PointerType *ins = static_cast<PointerType *>(create());
    ins->baseType = newBaseType;
    ins->variability = newVariability;
    return ins;
}
const PointerType *PointerType::createWithProperty(unsigned int newProperty) const {
    PointerType *ins = static_cast<PointerType *>(create());
    ins->property = newProperty;
    return ins;
}
const PointerType *PointerType::createWithConstAndProperty(bool newIsConst, unsigned int newProperty) const {
    PointerType *ins = static_cast<PointerType *>(create());
    ins->isConst = newIsConst;
    ins->property = newProperty;
    return ins;
}

PointerType *PointerType::GetUniform(const Type *t, bool is) {
    return new PointerType(t, Variability(Variability::Uniform), false, is ? SLICE : NONE);
}

PointerType *PointerType::GetVarying(const Type *t) {
    return new PointerType(t, Variability(Variability::Varying), false);
}

bool PointerType::IsVoidPointer(const Type *t) {
    return Type::EqualIgnoringConst(t->GetAsUniformType(), PointerType::Void);
}

const Type *PointerType::GetBaseType() const { return baseType; }

const PointerType *PointerType::GetAsSlice() const {
    if (IsSlice()) {
        return this;
    }
    return createWithProperty(SLICE);
}

const PointerType *PointerType::GetAsNonSlice() const {
    if (!IsSlice()) {
        return this;
    }
    return createWithProperty(NONE);
}

const PointerType *PointerType::GetAsFrozenSlice() const {
    if (IsFrozenSlice()) {
        return this;
    }
    return createWithProperty(FROZEN);
}

const PointerType *PointerType::GetWithAddrSpace(AddressSpace as) const {
    if (addrSpace == as) {
        return this;
    }
    return createWithAddressSpace(as);
}

const PointerType *PointerType::ResolveDependence(TemplateInstantiation &templInst) const {
    if (baseType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }

    const Type *resType = baseType->ResolveDependence(templInst);
    if (baseType == resType) {
        return this;
    }

    const PointerType *pType = createWithBaseType(resType);
    return pType;
}

const PointerType *PointerType::ResolveUnboundVariability(Variability v) const {
    if (baseType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }

    Assert(v != Variability::Unbound);
    Variability ptrVariability = (variability == Variability::Unbound) ? v : variability;
    const Type *resolvedBaseType = baseType->ResolveUnboundVariability(Variability::Uniform);
    return createWithBaseTypeAndVariability(resolvedBaseType, ptrVariability);
}

const PointerType *PointerType::GetAsConstType() const {
    if (isConst == true) {
        return this;
    } else {
        return createWithConstAndProperty(true, property & ~FROZEN);
    }
}

const PointerType *PointerType::GetAsNonConstType() const {
    if (isConst == false) {
        return this;
    } else {
        return createWithConstAndProperty(false, property & ~FROZEN);
    }
}

std::string PointerType::GetString() const {
    if (baseType == nullptr) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret = baseType->GetString();

    if (addrSpace != AddressSpace::ispc_default) {
        ret += std::string(" addrspace(") + std::to_string((int)addrSpace) + std::string(")");
    }

    ret += std::string(" * ");
    if (isConst) {
        ret += "const ";
    }
    if (IsSlice()) {
        ret += "slice ";
    }
    if (IsFrozenSlice()) {
        ret += "/*frozen*/ ";
    }
    ret += variability.GetString();

    return ret;
}

std::string PointerType::Mangle() const {
    Assert(variability != Variability::Unbound);
    if (baseType == nullptr) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret = variability.MangleString() + std::string("_3C_"); // <
    if (IsSlice() || IsFrozenSlice()) {
        ret += "-";
    }
    if (IsSlice()) {
        ret += "s";
    }
    if (IsFrozenSlice()) {
        ret += "f";
    }
    if (IsSlice() || IsFrozenSlice()) {
        ret += "-";
    }
    return ret + baseType->Mangle() + std::string("_3E_"); // >
}

std::string PointerType::GetDeclaration(const std::string &name, DeclarationSyntax syntax) const {
    if (IsSlice() || (variability == Variability::Unbound)) {
        Assert(m->errorCount > 0);
        return "";
    }

    if (baseType == nullptr) {
        Assert(m->errorCount > 0);
        return "";
    }

    bool baseIsBasicVarying = (IsBasicType(baseType)) && (baseType->IsVaryingType());
    bool baseIsFunction = (CastType<FunctionType>(baseType) != nullptr);

    std::string tempName;
    if (baseIsBasicVarying || baseIsFunction) {
        tempName += std::string("(");
    }
    tempName += std::string(" *");
    if (isConst) {
        tempName += " const";
    }
    tempName += std::string(" ");
    tempName += name;
    if (baseIsBasicVarying || baseIsFunction) {
        tempName += std::string(")");
    }

    std::string ret;
    if (!baseIsFunction) {
        ret = baseType->GetDeclaration("", syntax);
        ret += tempName;
    } else {
        ret += baseType->GetDeclaration(tempName, syntax);
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
    if (baseType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }

    if (IsSlice()) {
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
        llvm::Type *ptype = llvm::PointerType::get(*ctx, (unsigned)addrSpace);
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
        return nullptr;
    }
}

llvm::DIType *PointerType::GetDIType(llvm::DIScope *scope) const {
    if (baseType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    llvm::DIType *diTargetType = baseType->GetDIType(scope);
    int bitsSize = g->target->is32Bit() ? 32 : 64;
    int ptrAlignBits = bitsSize;
    switch (variability.type) {
    case Variability::Uniform:
        // Specifying address space for Xe target is necessary for correct work of SPIR-V Translator and other SPIR-V
        // tools like spirv-dis. If it's not specified the pointer storage class will be invalid, for example:
        // "DebugTypePointer 22 4294967295 0". In such case the SPIR-V tools may fail to disassemble it and return an
        // error: "221: Invalid storage class operand: 4294967295". What we really need to have here is one of the valid
        // storage classes: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_storage_class. For example:
        // "DebugTypePointer 167 7 0"
        return g->target->isXeTarget()
                   ? m->diBuilder->createPointerType(diTargetType, bitsSize, ptrAlignBits, (unsigned)addrSpace)
                   : m->diBuilder->createPointerType(diTargetType, bitsSize, ptrAlignBits);
    case Variability::Varying: {
        // emit them as an array of pointers
        llvm::DIDerivedType *eltType =
            g->target->isXeTarget()
                ? m->diBuilder->createPointerType(diTargetType, bitsSize, ptrAlignBits, (unsigned)addrSpace)
                : m->diBuilder->createPointerType(diTargetType, bitsSize, ptrAlignBits);
        return lCreateDIArray(eltType, g->target->getVectorWidth());
    }
    case Variability::SOA: {
        ArrayType at(GetAsUniformType(), variability.soaWidth);
        return at.GetDIType(scope);
    }
    default:
        FATAL("Unexpected variability in PointerType::GetDIType()");
        return nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////
// SequentialType

SequentialType::SequentialType(TypeId id, const Type *b, ElementCount ec, Variability v, bool c, SourcePos pos,
                               unsigned int al)
    : CollectionType(id, v, c, pos, al), base(b), elementCount(ec) {}

const SequentialType *SequentialType::createWithBaseType(const Type *newBaseType) const {
    SequentialType *ins = static_cast<SequentialType *>(create());
    ins->base = newBaseType;
    // Variability and constness of the SequentialType is derived from the base type, update it.
    ins->variability = newBaseType->GetVariability();
    ins->isConst = newBaseType->IsConstType();
    return ins;
}

const SequentialType *SequentialType::createWithElementCount(ElementCount newElementCount) const {
    SequentialType *ins = static_cast<SequentialType *>(create());
    ins->elementCount = newElementCount;
    return ins;
}

const SequentialType *SequentialType::createWithBaseTypeAndElementCount(const Type *newBaseType,
                                                                        ElementCount newElementCount) const {
    SequentialType *ins = static_cast<SequentialType *>(create());
    ins->base = newBaseType;
    // Variability and constness of the SequentialType is derived from the base type, update it.
    ins->variability = newBaseType->GetVariability();
    ins->isConst = newBaseType->IsConstType();
    ins->elementCount = newElementCount;
    return ins;
}

bool SequentialType::IsCompleteType() const { return base->IsCompleteType(); }

const SequentialType *SequentialType::GetAsVaryingType() const {
    if (IsVaryingType()) {
        return this;
    }
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsVaryingType());
}

const SequentialType *SequentialType::GetAsUniformType() const {
    if (IsUniformType()) {
        return this;
    }
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsUniformType());
}

const SequentialType *SequentialType::GetAsUnboundVariabilityType() const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsUnboundVariabilityType());
}

const SequentialType *SequentialType::GetAsSOAType(int width) const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsSOAType(width));
}

const SequentialType *SequentialType::ResolveUnboundVariability(Variability v) const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->ResolveUnboundVariability(v));
}

const SequentialType *SequentialType::GetAsUnsignedType() const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsUnsignedType());
}

const SequentialType *SequentialType::GetAsSignedType() const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsSignedType());
}

const SequentialType *SequentialType::GetAsConstType() const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsConstType());
}

const SequentialType *SequentialType::GetAsNonConstType() const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(base->GetAsNonConstType());
}

///////////////////////////////////////////////////////////////////////////
// ArrayType

// TODO!: Why ArrayType misses SourcePos??
ArrayType::ArrayType(const Type *b, int a)
    : SequentialType(ARRAY_TYPE, b, ElementCount(a), Variability::Uniform, false) {
    // 0 -> unsized array.
    Assert(elementCount.fixedCount >= 0);
    Assert(base->IsVoidType() == false);
    if (base) {
        variability = base->GetVariability();
        isConst = base->IsConstType();
    }
}

ArrayType::ArrayType(const Type *b, Symbol *num)
    : SequentialType(ARRAY_TYPE, b, ElementCount(num), Variability::Uniform, false) {
    if (base) {
        variability = base->GetVariability();
        isConst = base->IsConstType();
    }
}

ArrayType::ArrayType(const Type *b, ElementCount elCount)
    : SequentialType(ARRAY_TYPE, b, elCount, Variability::Uniform, false) {
    if (base) {
        variability = base->GetVariability();
        isConst = base->IsConstType();
    }
}

ArrayType::ArrayType(const ArrayType &other)
    : SequentialType(ARRAY_TYPE, other.base, other.elementCount, other.variability, other.isConst, other.pos,
                     other.alignment) {}

ArrayType *ArrayType::create() const { return new ArrayType(*this); }

llvm::ArrayType *ArrayType::LLVMType(llvm::LLVMContext *ctx) const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }

    llvm::Type *ct = base->LLVMStorageType(ctx);
    if (ct == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return llvm::ArrayType::get(ct, elementCount.fixedCount);
}

bool ArrayType::IsCompleteType() const { return GetBaseType()->IsCompleteType(); }

const Type *ArrayType::GetBaseType() const {
    const Type *type = base;
    const ArrayType *at = CastType<ArrayType>(type);
    // Keep walking until we reach a base that isn't itself an array
    while (at) {
        type = at->base;
        at = CastType<ArrayType>(type);
    }
    return type;
}

const ArrayType *ArrayType::ResolveDependence(TemplateInstantiation &templInst) const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    int resolvedCount = ResolveElementCount(templInst);
    const Type *resType = base->ResolveDependence(templInst);
    if (resType == base && resolvedCount == elementCount.fixedCount) {
        return this;
    }
    // TODO!: if element.symbolCount is 0 ?
    ElementCount elemCount = resolvedCount > 0 ? ElementCount(resolvedCount) : ElementCount(elementCount.symbolCount);
    return CastType<ArrayType>(createWithBaseTypeAndElementCount(resType, elemCount));
}

std::string ArrayType::GetString() const {
    const Type *base = GetBaseType();
    if (base == nullptr) {
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
        if (at->elementCount.fixedCount > 0) {
            snprintf(buf, sizeof(buf), "%d", at->elementCount.fixedCount);
        } else if (at->elementCount.symbolCount != nullptr) {
            snprintf(buf, sizeof(buf), "%s", at->elementCount.symbolCount->name.c_str());
        } else {
            buf[0] = '\0';
        }
        s += std::string("[") + std::string(buf) + std::string("]");
        at = CastType<ArrayType>(at->base);
    }
    return s;
}

std::string ArrayType::Mangle() const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return "(error)";
    }
    std::string s = base->Mangle();
    char buf[16];
    if (elementCount.fixedCount > 0) {
        snprintf(buf, sizeof(buf), "%d", elementCount.fixedCount);
    } else {
        buf[0] = '\0';
    }
    //    return s + "[" + buf + "]";
    return s + "_5B_" + buf + "_5D_";
}

std::string ArrayType::GetDeclaration(const std::string &name, DeclarationSyntax syntax) const {
    const Type *base = GetBaseType();
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return "";
    }

    int soaWidth = base->GetSOAWidth();
    int vWidth = (base->IsVaryingType()) ? g->target->getVectorWidth() : 0;
    base = base->GetAsUniformType();

    std::string s = base->GetDeclaration(name, syntax);

    const ArrayType *at = this;
    Assert(at);
    while (at) {
        char buf[16];
        if (at->elementCount.fixedCount > 0) {
            snprintf(buf, sizeof(buf), "%d", at->elementCount.fixedCount);
        } else {
            buf[0] = '\0';
        }
        s += std::string("[") + std::string(buf) + std::string("]");
        at = CastType<ArrayType>(at->base);
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
    const ArrayType *ct = CastType<ArrayType>(base);
    if (ct != nullptr) {
        return elementCount.fixedCount * ct->TotalElementCount();
    } else {
        return elementCount.fixedCount;
    }
}

llvm::DIType *ArrayType::GetDIType(llvm::DIScope *scope) const {
    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    llvm::DIType *eltType = base->GetDIType(scope);
    return lCreateDIArray(eltType, elementCount.fixedCount);
}

bool ArrayType::IsUnsized() const { return (elementCount.fixedCount == 0 && elementCount.symbolCount == nullptr); }

const ArrayType *ArrayType::GetSizedArray(int sz) const {
    Assert(elementCount.fixedCount == 0);
    // TODO!: if sz is 0, we should issue an error here?
    return CastType<ArrayType>(createWithElementCount(ElementCount(sz)));
}

const Type *ArrayType::SizeUnsizedArrays(const Type *type, Expr *initExpr) {
    const ArrayType *at = CastType<ArrayType>(type);
    if (at == nullptr) {
        return type;
    }

    ExprList *exprList = llvm::dyn_cast_or_null<ExprList>(initExpr);
    if (exprList == nullptr || exprList->exprs.size() == 0) {
        return type;
    }

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
    if (nextList == nullptr) {
        return type;
    }

    Assert(at);
    const Type *nextType = at->GetElementType();
    const ArrayType *nextArrayType = CastType<ArrayType>(nextType);
    if (nextArrayType != nullptr && nextArrayType->GetElementCount() == 0) {
        // If the recursive call to SizeUnsizedArrays at the bottom of the
        // function is going to size an unsized dimension, make sure that
        // all of the sub-expression lists are the same length--i.e. issue
        // an error if we have something like
        // int x[][] = { { 1 }, { 1, 2, 3, 4 } };
        unsigned int nextSize = nextList->exprs.size();
        for (unsigned int i = 1; i < exprList->exprs.size(); ++i) {
            if (exprList->exprs[i] == nullptr) {
                // We should have seen an error earlier in this case.
                Assert(m->errorCount > 0);
                continue;
            }

            ExprList *el = llvm::dyn_cast_or_null<ExprList>(exprList->exprs[i]);
            if (el == nullptr || el->exprs.size() != nextSize) {
                Error(Union(exprList->exprs[0]->pos, exprList->exprs[i]->pos),
                      "Inconsistent initializer expression list lengths "
                      "make it impossible to size unsized array dimensions.");
                return nullptr;
            }
        }
    }

    // Recursively call SizeUnsizedArrays() to get the base type for the
    // array that we were able to size here.
    return new ArrayType(SizeUnsizedArrays(at->GetElementType(), nextList), at->GetElementCount());
}

int ArrayType::ResolveElementCount(TemplateInstantiation &templInst) const {
    if (elementCount.fixedCount > 0) {
        return elementCount.fixedCount;
    }
    if (elementCount.symbolCount) {
        Symbol *instSym = templInst.InstantiateSymbol(elementCount.symbolCount);
        if (!instSym->constValue) {
            // failed to resolve constant value, return elementCount.fixedCount
            return elementCount.fixedCount;
        }
        unsigned int constValue[1];
        int count = instSym->constValue->GetValues(constValue);
        return (count > 0) ? constValue[0] : 0;
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////
// VectorType

VectorType::VectorType(const Type *b, int a)
    : SequentialType(VECTOR_TYPE, b, ElementCount(a), b->GetVariability(), b->IsConstType()) {
    Assert(elementCount.fixedCount > 0);
}

VectorType::VectorType(const Type *b, Symbol *num)
    : SequentialType(VECTOR_TYPE, b, ElementCount(num), b->GetVariability(), b->IsConstType()) {}

VectorType::VectorType(const Type *b, ElementCount elCount)
    : SequentialType(VECTOR_TYPE, b, elCount, b->GetVariability(), b->IsConstType()) {}

VectorType::VectorType(const VectorType &other)
    : SequentialType(VECTOR_TYPE, other.base, other.elementCount, other.variability, other.isConst, other.pos,
                     other.alignment) {}

VectorType *VectorType::create() const { return new VectorType(*this); }

bool VectorType::IsFloatType() const { return base->IsFloatType(); }

bool VectorType::IsIntType() const { return base->IsIntType(); }

bool VectorType::IsUnsignedType() const { return base->IsUnsignedType(); }

bool VectorType::IsSignedType() const { return base->IsSignedType(); }

bool VectorType::IsBoolType() const { return base->IsBoolType(); }

const VectorType *VectorType::ResolveDependence(TemplateInstantiation &templInst) const {
    int resolvedCount = ResolveElementCount(templInst);
    const Type *resolvedBase = base->ResolveDependence(templInst);
    ElementCount elemCount = (resolvedCount > 0) ? ElementCount(resolvedCount) : ElementCount(elementCount.symbolCount);
    return CastType<VectorType>(createWithBaseTypeAndElementCount(resolvedBase, elemCount));
}

std::string VectorType::GetCountString() const {
    char buf[16];
    if (elementCount.fixedCount > 0 || elementCount.symbolCount == nullptr) {
        snprintf(buf, sizeof(buf), "%d", elementCount.fixedCount);
    } else {
        snprintf(buf, sizeof(buf), "%s", elementCount.symbolCount->name.c_str());
    }
    return std::string(buf);
}

std::string VectorType::GetString() const {
    return base->GetString() + std::string("<") + GetCountString() + std::string(">");
}

std::string VectorType::Mangle() const {
    return base->Mangle() + std::string("_3C_") + GetCountString() + std::string("_3E_");
}

std::string VectorType::GetDeclaration(const std::string &name, DeclarationSyntax syntax) const {
    return base->GetDeclaration("", syntax) + GetCountString() + "  " + name;
}

static llvm::Type *lGetVectorLLVMType(llvm::LLVMContext *ctx, const VectorType *vType, bool isStorage) {

    const Type *base = vType->GetBaseType();
    int numElements = vType->GetElementCount();

    if (base == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }

    llvm::Type *bt = nullptr;
    // Non-uniform vector types are represented in IR as an array.
    // So, creating them with base as storage type similar to arrays.
    if (isStorage || !base->IsUniformType()) {
        bt = base->LLVMStorageType(ctx);
    } else {
        bt = base->LLVMType(ctx);
    }
    if (!bt) {
        return nullptr;
    }

    if (base->IsUniformType()) {
        // Vectors of uniform types are laid out across LLVM vectors, with
        // the llvm vector size set to be a power of 2 bits in size but not less then 128 bit.
        // This is a roundabout way of ensuring that LLVM lays
        // them out into machine vector registers for the specified target
        // so that e.g. if we want to add two uniform 4 float
        // vectors, that is turned into a single addps on SSE.
        return LLVMVECTOR::get(bt, vType->getVectorMemoryCount());
    } else if (base->IsVaryingType()) {
        // varying types are already laid out to fill HW vector registers,
        // so a vector type here is just expanded out as an llvm array.
        return llvm::ArrayType::get(bt, vType->getVectorMemoryCount());
    } else if (base->IsSOAType()) {
        return llvm::ArrayType::get(bt, numElements);
    } else {
        FATAL("Unexpected variability in lGetVectorLLVMType()");
        return nullptr;
    }
}

llvm::Type *VectorType::LLVMStorageType(llvm::LLVMContext *ctx) const { return lGetVectorLLVMType(ctx, this, true); }

llvm::Type *VectorType::LLVMType(llvm::LLVMContext *ctx) const { return lGetVectorLLVMType(ctx, this, false); }

llvm::DIType *VectorType::GetDIType(llvm::DIScope *scope) const {
    llvm::DIType *eltType = base->GetDIType(scope);

    llvm::Metadata *sub = m->diBuilder->getOrCreateSubrange(0, elementCount.fixedCount);

    // vectors of varying types are already naturally aligned to the
    // machine's vector width, but arrays of uniform types need to be
    // explicitly aligned to the machines natural vector alignment.

    llvm::DINodeArray subArray = m->diBuilder->getOrCreateArray(sub);
    uint64_t sizeBits = eltType->getSizeInBits() * elementCount.fixedCount;
    uint64_t align = eltType->getAlignInBits();

    if (IsUniformType()) {
        llvm::Type *ty = this->LLVMType(g->ctx);
        align = g->target->getDataLayout()->getABITypeAlign(ty).value();
    }

    if (IsUniformType() || IsVaryingType()) {
        return m->diBuilder->createVectorType(sizeBits, align, eltType, subArray);
    } else if (IsSOAType()) {
        ArrayType at(base, elementCount.fixedCount);
        return at.GetDIType(scope);
    } else {
        FATAL("Unexpected variability in VectorType::GetDIType()");
        return nullptr;
    }
}

int VectorType::getVectorMemoryCount() const {
    if (base->IsVaryingType()) {
        return elementCount.fixedCount;
    } else if (base->IsUniformType()) {
        // Round up the element count to power of 2 bits in size but not less then 128 bit in total vector size
        // where one element size is data type width in bits.
        // This strategy was chosen by the following reasons:
        // 1. We need to return the same number of vector elements regardless the element size for correct work of the
        // language.
        // 2. Using next power of two, but not less than 128 bit in total vector size ensures that machine vector
        // registers are used. It generally leads to better performance. This strategy also matches OpenCL short
        // vectors.
        // 3. Using data type width of the target to determine element size makes optimization trade off.
        int nextPow2 = llvm::NextPowerOf2(elementCount.fixedCount - 1);
        return (nextPow2 * g->target->getDataTypeWidth() < 128) ? (128 / g->target->getDataTypeWidth()) : nextPow2;
    } else if (base->IsSOAType()) {
        FATAL("VectorType SOA getVectorMemoryCount");
        return -1;
    } else {
        FATAL("Unexpected variability in VectorType::getVectorMemoryCount()");
        return -1;
    }
}

int VectorType::ResolveElementCount(TemplateInstantiation &templInst) const {
    if (elementCount.fixedCount > 0) {
        return elementCount.fixedCount;
    }
    if (elementCount.symbolCount) {
        Symbol *instSym = templInst.InstantiateSymbol(elementCount.symbolCount);
        if (!instSym->constValue) {
            // failed to resolve constant value, return elementCount.fixedCount
            return elementCount.fixedCount;
        }
        unsigned int constValue[1];
        int count = instSym->constValue->GetValues(constValue);
        return (count > 0) ? constValue[0] : 0;
    }
    return 0;
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
                       Variability v, SourcePos p, unsigned int align)
    : CollectionType(STRUCT_TYPE, v, ic, p, align), name(n), elementTypes(elts), elementNames(en), elementPositions(ep),
      isAnonymous(name.empty()) {
    oppositeConstStructType = nullptr;
    finalElementTypes.resize(elts.size(), nullptr);
}

void StructType::RegisterInStructTypeMap() {
    static int count = 0;
    // Create a unique anonymous struct name if we have an anonymous struct (name == "")
    // Ensuring struct is created with a name, prevents each use of original
    // struct from having different names causing type match errors.
    if (name == "") {
        name = "$anon" + std::to_string(count);
        ++count;
    }

    if (variability != Variability::Unbound) {
        // For structs with non-unbound variability, we'll create the
        // correspoing LLVM struct type now, if one hasn't been made
        // already.

        // If a non-opaque LLVM struct for this type has already been
        // created, we're done.  For an opaque struct type, we'll override
        // the old definition now that we have a full definition.
        std::string mname = lMangleStructName(name, variability);
        if (m->structTypeMap.find(mname) != m->structTypeMap.end() && m->structTypeMap[mname]->isOpaque() == false) {
            return;
        }

        // Actually make the LLVM struct
        std::vector<llvm::Type *> elementTypes;
        int nElements = GetElementCount();
        if (nElements == 0) {
            elementTypes.push_back(LLVMTypes::Int8Type);
        } else {
            for (int i = 0; i < nElements; ++i) {
                const Type *type = GetElementType(i);
                if (type == nullptr) {
                    Assert(m->errorCount > 0);
                    return;
                } else if (CastType<FunctionType>(type) != nullptr) {
                    Error(elementPositions[i], "Method declarations are not "
                                               "supported.");
                    return;
                } else {
                    elementTypes.push_back(type->LLVMStorageType(g->ctx));
                }
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
}

StructType::StructType(const StructType &other)
    : CollectionType(STRUCT_TYPE, other.variability, other.isConst, other.pos, other.alignment), name(other.name),
      elementTypes(other.elementTypes), elementNames(other.elementNames), elementPositions(other.elementPositions),
      isAnonymous(other.isAnonymous) {
    oppositeConstStructType = nullptr;
    finalElementTypes.resize(other.elementTypes.size(), nullptr);
}

StructType *StructType::create() const { return new StructType(*this); }

const StructType *StructType::createWithVariability(Variability newVariability) const {
    StructType *t = create();
    t->variability = newVariability;
    t->RegisterInStructTypeMap();
    return t;
}

const StructType *StructType::createWithConst(bool newIsConst) const {
    StructType *t = create();
    t->isConst = newIsConst;
    t->RegisterInStructTypeMap();
    return t;
}

const StructType *StructType::createWithAlignment(unsigned int newAlignment) const {
    StructType *t = create();
    t->alignment = newAlignment;
    t->RegisterInStructTypeMap();
    return t;
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

bool StructType::IsCompleteType() const {
    int n = GetElementCount();
    // Corner case when struct consists of a single unsized array
    if (n == 1) {
        if (const ArrayType *arrayType = CastType<ArrayType>(GetRawElementType(0))) {
            return !arrayType->IsUnsized() && arrayType->IsCompleteType();
        }
    }

    // Then check that all elements are complete types
    for (int i = 0; i < n; i++) {
        if (const Type *ET = GetRawElementType(i)) {
            if (!ET->IsCompleteType()) {
                return false;
            }
        }
    }
    return true;
}

bool StructType::IsDefined() const {
    for (int i = 0; i < GetElementCount(); i++) {
        const Type *t = GetElementType(i);
        const UndefinedStructType *ust = CastType<UndefinedStructType>(t);
        if (ust != nullptr) {
            return false;
        }
        const StructType *st = CastType<StructType>(t);
        if (st != nullptr) {
            if (!st->IsDefined()) {
                return false;
            }
        }
    }
    return true;
}

bool StructType::IsAnonymousType() const { return isAnonymous; }

const StructType *StructType::GetAsSOAType(int width) const {
    if (GetSOAWidth() == width) {
        return this;
    }

    if (checkIfCanBeSOA(this) == false) {
        return nullptr;
    }

    return createWithVariability(Variability(Variability::SOA, width));
}

const StructType *StructType::GetAsNamed(const std::string &n) const {
    Assert(n != "" && "We should not create anonymous structs here.");
    StructType *t = create();
    t->name = n;
    t->isAnonymous = false;
    t->RegisterInStructTypeMap();
    return t;
}

std::string StructType::GetString() const {
    std::string ret;
    if (isConst) {
        ret += "const ";
    }
    ret += variability.GetString();
    ret += " struct ";

    if (IsAnonymousType()) {
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
    if (isConst) {
        ret += "_c_";
    }
    ret += variability.MangleString();

    //    ret += name + std::string("]");
    ret += name + std::string("_5D_");
    return ret;
}

std::string StructType::Mangle() const { return lMangleStruct(variability, isConst, name); }

std::string StructType::GetDeclaration(const std::string &n, DeclarationSyntax syntax) const {
    std::string ret;
    if (isConst) {
        ret += "const ";
    }
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
        return nullptr;
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
        uint64_t eltAlign = dataLayout.getABITypeAlign(llvmType).value() * 8;
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
                                          nullptr, elements);
}

const Type *StructType::GetElementType(int i) const {
    Assert(variability != Variability::Unbound);
    Assert(i < (int)elementTypes.size());

    if (finalElementTypes[i] == nullptr) {
        const Type *type = elementTypes[i];
        if (type == nullptr) {
            Assert(m->errorCount > 0);
            return nullptr;
        }

        // If the element has unbound variability, resolve its variability to
        // the struct type's variability
        type = type->ResolveUnboundVariability(variability);
        if (isConst) {
            type = type->GetAsConstType();
        }
        finalElementTypes[i] = type;
    }

    return finalElementTypes[i];
}

const Type *StructType::GetRawElementType(int i) const {
    Assert(i < (int)elementTypes.size());
    return elementTypes[i];
}

const Type *StructType::GetElementType(const std::string &n) const {
    for (unsigned int i = 0; i < elementNames.size(); ++i) {
        if (elementNames[i] == n) {
            return GetElementType(i);
        }
    }
    return nullptr;
}

int StructType::GetElementNumber(const std::string &n) const {
    for (unsigned int i = 0; i < elementNames.size(); ++i) {
        if (elementNames[i] == n) {
            return i;
        }
    }
    return -1;
}

bool StructType::checkIfCanBeSOA(const StructType *st) {
    bool ok = true;
    for (int i = 0; i < (int)st->elementTypes.size(); ++i) {
        const Type *eltType = st->elementTypes[i];
        const StructType *childStructType = CastType<StructType>(eltType);

        if (childStructType != nullptr) {
            ok &= checkIfCanBeSOA(childStructType);
        } else if (eltType->HasUnboundVariability() == false) {
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

UndefinedStructType::UndefinedStructType(const std::string &n, const Variability var, bool ic, SourcePos p,
                                         unsigned int align)
    : Type(UNDEFINED_STRUCT_TYPE, var, ic, p, align), name(n) {
    Assert(name != "");
}

UndefinedStructType::UndefinedStructType(const UndefinedStructType &other)
    : Type(UNDEFINED_STRUCT_TYPE, other.variability, other.isConst, other.pos, other.alignment), name(other.name) {}

void UndefinedStructType::RegisterInStructTypeMap() {
    if (variability != Variability::Unbound) {
        // Create a new opaque LLVM struct type for this struct name
        std::string mname = lMangleStructName(name, variability);
        if (m->structTypeMap.find(mname) == m->structTypeMap.end()) {
            m->structTypeMap[mname] = llvm::StructType::create(*g->ctx, mname);
        }
    }
}

UndefinedStructType *UndefinedStructType::create() const { return new UndefinedStructType(*this); }

const UndefinedStructType *UndefinedStructType::createWithVariability(Variability newVariability) const {
    UndefinedStructType *t = create();
    t->variability = newVariability;
    t->RegisterInStructTypeMap();
    return t;
}

const UndefinedStructType *UndefinedStructType::createWithConst(bool newIsConst) const {
    UndefinedStructType *t = create();
    t->isConst = newIsConst;
    t->RegisterInStructTypeMap();
    return t;
}

const UndefinedStructType *UndefinedStructType::createWithAlignment(unsigned int newAlignment) const {
    UndefinedStructType *t = create();
    t->alignment = newAlignment;
    t->RegisterInStructTypeMap();
    return t;
}

bool UndefinedStructType::IsCompleteType() const { return false; }

const UndefinedStructType *UndefinedStructType::GetAsSOAType(int width) const {
    FATAL("UndefinedStructType::GetAsSOAType() shouldn't be called.");
    return nullptr;
}

std::string UndefinedStructType::GetString() const {
    std::string ret;
    if (isConst) {
        ret += "const ";
    }
    ret += variability.GetString();
    ret += " struct ";
    ret += name;
    return ret;
}

std::string UndefinedStructType::Mangle() const { return lMangleStruct(variability, isConst, name); }

std::string UndefinedStructType::GetDeclaration(const std::string &n, DeclarationSyntax syntax) const {
    std::string ret;
    if (isConst) {
        ret += "const ";
    }
    ret += std::string("struct ") + name;
    if (lShouldPrintName(n)) {
        ret += std::string(" ") + n;
    }
    return ret;
}

llvm::Type *UndefinedStructType::LLVMType(llvm::LLVMContext *ctx) const {
    Assert(variability != Variability::Unbound);
    std::string mname = lMangleStructName(name, variability);
    if (m->structTypeMap.find(mname) == m->structTypeMap.end()) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return m->structTypeMap[mname];
}

llvm::DIType *UndefinedStructType::GetDIType(llvm::DIScope *scope) const {
    llvm::DIFile *diFile = pos.GetDIFile();
    llvm::DINamespace *diSpace = pos.GetDINamespace();
    llvm::DINodeArray elements;
    uint64_t sizeInBits = 0;
    return m->diBuilder->createStructType(diSpace, GetString(), diFile,
                                          pos.first_line,         // Line number
                                          sizeInBits,             // Size
                                          0,                      // Align
                                          llvm::DINode::FlagZero, // Flags
                                          nullptr, elements);
}

///////////////////////////////////////////////////////////////////////////
// ReferenceType

ReferenceType::ReferenceType(const Type *t, AddressSpace as)
    : Type(REFERENCE_TYPE, Variability::Unbound, false), targetType(t), addrSpace(as) {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
    } else {
        variability = targetType->GetVariability();
        isConst = targetType->IsConstType();
        asOtherConstType = nullptr;
    }
}

ReferenceType::ReferenceType(const ReferenceType &other) : ReferenceType(other.targetType, other.addrSpace) {}

ReferenceType *ReferenceType::create() const { return new ReferenceType(*this); }

const ReferenceType *ReferenceType::createWithBaseType(const Type *newBaseType) const {
    ReferenceType *ins = static_cast<ReferenceType *>(create());
    ins->targetType = newBaseType;
    return ins;
}
const ReferenceType *ReferenceType::createWithAddressSpace(AddressSpace newAddrSpace) const {
    ReferenceType *ins = static_cast<ReferenceType *>(create());
    ins->addrSpace = newAddrSpace;
    return ins;
}

bool ReferenceType::IsBoolType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsBoolType();
}

bool ReferenceType::IsFloatType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsFloatType();
}

bool ReferenceType::IsIntType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsIntType();
}

bool ReferenceType::IsUnsignedType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsUnsignedType();
}

bool ReferenceType::IsSignedType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return false;
    }
    return targetType->IsSignedType();
}

const Type *ReferenceType::GetReferenceTarget() const { return targetType; }

const Type *ReferenceType::GetBaseType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return targetType->GetBaseType();
}

const ReferenceType *ReferenceType::GetAsVaryingType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    if (IsVaryingType()) {
        return this;
    }
    return createWithBaseType(targetType->GetAsVaryingType());
}

const ReferenceType *ReferenceType::GetAsUniformType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    if (IsUniformType()) {
        return this;
    }
    return createWithBaseType(targetType->GetAsUniformType());
}

const ReferenceType *ReferenceType::GetAsUnboundVariabilityType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    if (HasUnboundVariability()) {
        return this;
    }
    return createWithBaseType(targetType->GetAsUnboundVariabilityType());
}

const Type *ReferenceType::GetAsSOAType(int width) const {
    // FIXME: is this right?
    return new ArrayType(this, width);
}

const ReferenceType *ReferenceType::ResolveDependence(TemplateInstantiation &templInst) const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(targetType->ResolveDependence(templInst));
}

const ReferenceType *ReferenceType::ResolveUnboundVariability(Variability v) const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return createWithBaseType(targetType->ResolveUnboundVariability(v));
}

const ReferenceType *ReferenceType::GetAsConstType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    if (IsConstType()) {
        return this;
    }

    if (asOtherConstType == nullptr) {
        const ReferenceType *t = createWithBaseType(targetType->GetAsConstType());
        t->asOtherConstType = this;
        asOtherConstType = t;
    }
    return CastType<ReferenceType>(asOtherConstType);
}

const ReferenceType *ReferenceType::GetAsNonConstType() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    if (!IsConstType()) {
        return this;
    }

    if (asOtherConstType == nullptr) {
        const ReferenceType *t = createWithBaseType(targetType->GetAsNonConstType());
        t->asOtherConstType = this;
        asOtherConstType = t;
    }
    return CastType<ReferenceType>(asOtherConstType);
}

const ReferenceType *ReferenceType::GetWithAddrSpace(AddressSpace as) const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    if (addrSpace == as) {
        return this;
    }

    return createWithAddressSpace(as);
}

std::string ReferenceType::GetString() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return "";
    }

    std::string ret = targetType->GetString();

    if (addrSpace != AddressSpace::ispc_default) {
        ret += std::string(" addrspace(") + std::to_string((int)addrSpace) + std::string(")");
    }

    ret += std::string(" &");
    return ret;
}

std::string ReferenceType::Mangle() const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return "";
    }
    std::string ret;
    ret += std::string("REF") + targetType->Mangle();
    return ret;
}

std::string ReferenceType::GetDeclaration(const std::string &name, DeclarationSyntax syntax) const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return "";
    }

    const ArrayType *at = CastType<ArrayType>(targetType);
    if (at != nullptr) {
        if (at->GetElementCount() == 0) {
            // emit unsized arrays as pointers to the base type..
            std::string ret;
            ret += at->GetElementType()->GetAsNonConstType()->GetDeclaration("", syntax) + std::string(" *");
            if (lShouldPrintName(name)) {
                ret += name;
            }
            return ret;
        } else {
            // otherwise forget about the reference part if it's an
            // array since C already passes arrays by reference...
            return targetType->GetDeclaration(name, syntax);
        }
    } else {
        std::string ret;
        ret += targetType->GetDeclaration("", syntax);
        ret += syntax == DeclarationSyntax::CPP ? std::string(" &") : std::string(" *");
        if (lShouldPrintName(name)) {
            ret += name;
        }
        return ret;
    }
}

llvm::Type *ReferenceType::LLVMType(llvm::LLVMContext *ctx) const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    return llvm::PointerType::get(*ctx, (unsigned)addrSpace);
}

llvm::DIType *ReferenceType::GetDIType(llvm::DIScope *scope) const {
    if (targetType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    llvm::DIType *diTargetType = targetType->GetDIType(scope);
    // Specifying address space for Xe target is necessary for correct work of SPIR-V Translator and other SPIR-V
    // tools like spirv-dis. See more detailed description in PointerType::GetDIType
    return g->target->isXeTarget()
               ? m->diBuilder->createReferenceType(llvm::dwarf::DW_TAG_reference_type, diTargetType, 0, 0,
                                                   (unsigned)addrSpace)
               : m->diBuilder->createReferenceType(llvm::dwarf::DW_TAG_reference_type, diTargetType);
}

///////////////////////////////////////////////////////////////////////////
// FunctionType

FunctionType::FunctionType(const Type *r, const llvm::SmallVector<const Type *, 8> &a, SourcePos p)
    : Type(FUNCTION_TYPE, Variability::Uniform, false, p), returnType(r), paramTypes(a),
      paramNames(llvm::SmallVector<std::string, 8>(a.size(), "")),
      paramDefaults(llvm::SmallVector<Expr *, 8>(a.size(), nullptr)),
      paramPositions(llvm::SmallVector<SourcePos, 8>(a.size(), p)), flags(0) {
    Assert(returnType != nullptr);
    costOverride = -1;
    asUnmaskedType = asMaskedType = nullptr;
}

FunctionType::FunctionType(const Type *r, const llvm::SmallVector<const Type *, 8> &a,
                           const llvm::SmallVector<std::string, 8> &an, const llvm::SmallVector<Expr *, 8> &ad,
                           const llvm::SmallVector<SourcePos, 8> &ap, int cost, unsigned int functionFlags, SourcePos p)
    : Type(FUNCTION_TYPE, Variability::Uniform, false, p), returnType(r), paramTypes(a), paramNames(an),
      paramDefaults(ad), paramPositions(ap), costOverride(cost), flags(functionFlags) {
    Assert(paramTypes.size() == paramNames.size() && paramNames.size() == paramDefaults.size() &&
           paramDefaults.size() == paramPositions.size());
    Assert(returnType != nullptr);
    asUnmaskedType = asMaskedType = nullptr;
}

FunctionType::FunctionType(const FunctionType &other)
    : FunctionType(other.returnType, other.paramTypes, other.paramNames, other.paramDefaults, other.paramPositions,
                   other.costOverride, other.flags, other.pos) {
    // Reset any cached values
    this->asMaskedType = nullptr;
    this->asUnmaskedType = nullptr;
}

FunctionType *FunctionType::create() const { return new FunctionType(*this); }

const FunctionType *FunctionType::createWithSignature(const Type *newReturnType,
                                                      const llvm::SmallVector<const Type *, 8> &newParamTypes) const {
    FunctionType *ins = static_cast<FunctionType *>(create());
    ins->returnType = newReturnType;
    ins->paramTypes = newParamTypes;
    return ins;
}

const FunctionType *FunctionType::createWithFlags(unsigned int newFlags) const {
    FunctionType *ins = static_cast<FunctionType *>(create());
    ins->flags = newFlags;
    return ins;
}

bool FunctionType::IsISPCKernel() const { return g->target->isXeTarget() && IsTask(); }

bool FunctionType::IsISPCExternal() const {
    return g->target->isXeTarget() && (IsExported() || IsExternC() || IsExternSYCL());
}

const Type *FunctionType::GetBaseType() const {
    FATAL("FunctionType::GetBaseType() shouldn't be called");
    return nullptr;
}

const Type *FunctionType::GetAsVaryingType() const { return nullptr; }

const Type *FunctionType::GetAsUniformType() const {
    // According to FunctionType::GetVariability(), FunctionType is always
    // uniform, so it is safe to return this here.
    return this;
}

const Type *FunctionType::GetAsUnboundVariabilityType() const {
    FATAL("FunctionType::GetAsUnboundVariabilityType shouldn't be called");
    return nullptr;
}

const Type *FunctionType::GetAsSOAType(int width) const {
    FATAL("FunctionType::GetAsSOAType() shouldn't be called");
    return nullptr;
}

const FunctionType *FunctionType::ResolveDependence(TemplateInstantiation &templInst) const {
    if (returnType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    const Type *rt = returnType->ResolveDependenceForTopType(templInst);

    llvm::SmallVector<const Type *, 8> pt;
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == nullptr) {
            Assert(m->errorCount > 0);
            return nullptr;
        }
        const Type *argt = paramTypes[i]->ResolveDependenceForTopType(templInst);
        pt.push_back(argt);
    }

    return createWithSignature(rt, pt);
}

const FunctionType *FunctionType::ResolveUnboundVariability(Variability v) const {
    if (returnType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }
    const Type *rt = returnType->ResolveUnboundVariability(v);

    llvm::SmallVector<const Type *, 8> pt;
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == nullptr) {
            Assert(m->errorCount > 0);
            return nullptr;
        }
        pt.push_back(paramTypes[i]->ResolveUnboundVariability(v));
    }

    return createWithSignature(rt, pt);
}

const Type *FunctionType::GetAsConstType() const { return this; }

const Type *FunctionType::GetAsNonConstType() const { return this; }

const Type *FunctionType::GetAsUnmaskedType() const {
    if (IsUnmasked()) {
        return this;
    }
    if (asUnmaskedType == nullptr) {
        asUnmaskedType = createWithFlags(flags | FUNC_UNMASKED);
        if (!IsUnmasked()) {
            asUnmaskedType->asMaskedType = this;
        }
    }
    return asUnmaskedType;
}

const Type *FunctionType::GetAsNonUnmaskedType() const {
    if (!IsUnmasked()) {
        return this;
    }
    if (asMaskedType == nullptr) {
        asMaskedType = createWithFlags(flags & ~FUNC_UNMASKED);
        if (IsUnmasked()) {
            asMaskedType->asUnmaskedType = this;
        }
    }
    return asMaskedType;
}

const Type *FunctionType::GetWithReturnType(const Type *newReturnType) const {
    if (returnType == newReturnType) {
        return this;
    }

    return createWithSignature(newReturnType, paramTypes);
}

std::string FunctionType::GetString() const {
    std::string ret = GetNameForCallConv();
    ret += " ";
    ret += GetReturnTypeString();
    ret += "(";
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == nullptr) {
            ret += "/* ERROR */";
        } else {
            ret += paramTypes[i]->GetString();
        }

        ret += " " + paramNames[i];

        if (paramDefaults[i] != nullptr) {
            ret += " = init";
        }

        if (i != paramTypes.size() - 1) {
            ret += ", ";
        }
    }
    ret += ")";
    return ret;
}

std::string FunctionType::Mangle() const {
    std::string ret = "___";
    if (IsUnmasked()) {
        ret += "UM_";
    }

    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == nullptr) {
            Assert(m->errorCount > 0);
        } else {
            ret += paramTypes[i]->Mangle();
        }
    }

    return ret;
}

std::string FunctionType::GetDeclaration(const std::string &fname, DeclarationSyntax syntax) const {
    std::string ret;
    ret += returnType->GetDeclaration("", syntax);
    ret += " ";
    ret += fname;
    ret += "(";
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        const Type *type = paramTypes[i];

        // Convert pointers to arrays to unsized arrays, which are more clear
        // to print out for multidimensional arrays (i.e. "float foo[][4] "
        // versus "float (foo *)[4]").
        const PointerType *pt = CastType<PointerType>(type);
        if (pt != nullptr && CastType<ArrayType>(pt->GetBaseType()) != nullptr) {
            type = new ArrayType(pt->GetBaseType(), 0);
        }

        if (paramNames[i] != "") {
            ret += type->GetDeclaration(paramNames[i], syntax);
        } else {
            ret += type->GetString();
        }
        if (i != paramTypes.size() - 1) {
            ret += ", ";
        }
    }
    ret += ")";
    return ret;
}

std::string FunctionType::GetDeclarationForDispatch(const std::string &fname, DeclarationSyntax syntax) const {
    std::string ret;
    ret += returnType->GetDeclaration("", syntax);
    ret += " ";
    ret += fname;
    ret += "(";
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        const Type *type = paramTypes[i];

        // Convert pointers to arrays to unsized arrays, which are more clear
        // to print out for multidimensional arrays (i.e. "float foo[][4] "
        // versus "float (foo *)[4]").
        const PointerType *pt = CastType<PointerType>(type);
        if (pt != nullptr && CastType<ArrayType>(pt->GetBaseType()) != nullptr) {
            type = new ArrayType(pt->GetBaseType(), 0);
        }

        // Change pointers to varying thingies to void *
        if (pt != nullptr && pt->GetBaseType()->IsVaryingType()) {
            PointerType *t = PointerType::Void;

            if (paramNames[i] != "") {
                ret += t->GetDeclaration(paramNames[i], syntax);
            } else {
                ret += t->GetString();
            }
        } else {
            if (paramNames[i] != "") {
                ret += type->GetDeclaration(paramNames[i], syntax);
            } else {
                ret += type->GetString();
            }
        }
        if (i != paramTypes.size() - 1) {
            ret += ", ";
        }
    }
    ret += ")";
    return ret;
}

llvm::Type *FunctionType::LLVMType(llvm::LLVMContext *ctx) const {
    FATAL("FunctionType::LLVMType() shouldn't be called");
    return nullptr;
}

llvm::DIType *FunctionType::GetDIType(llvm::DIScope *scope) const {

    std::vector<llvm::Metadata *> retArgTypes;
    retArgTypes.push_back(returnType->GetDIType(scope));
    for (int i = 0; i < GetNumParameters(); ++i) {
        const Type *t = GetParameterType(i);
        if (t == nullptr) {
            return nullptr;
        }
        retArgTypes.push_back(t->GetDIType(scope));
    }

    llvm::DITypeRefArray retArgTypesArray = m->diBuilder->getOrCreateTypeArray(retArgTypes);
    llvm::DIType *diType = m->diBuilder->createSubroutineType(retArgTypesArray);
    return diType;
}

const std::string FunctionType::GetReturnTypeString() const {
    if (returnType == nullptr) {
        return "/* ERROR */";
    }

    std::string ret;
    if (IsTask()) {
        ret += "task ";
    }
    if (IsExported()) {
        ret += "export ";
    }
    if (IsExternC()) {
        ret += "extern \"C\" ";
    }
    if (IsExternSYCL()) {
        ret += "extern \"SYCL\" ";
    }
    if (IsUnmasked()) {
        ret += "unmasked ";
    }
    if (IsSafe()) {
        ret += "/*safe*/ ";
    }
    if (costOverride > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "/*cost=%d*/ ", costOverride);
        ret += buf;
    }

    return ret + returnType->GetString();
}

std::string FunctionType::mangleTemplateArgs(TemplateArgs *templateArgs) const {
    if (templateArgs == nullptr) {
        return "";
    }
    std::string ret = "___";
    for (const auto &arg : *templateArgs) {
        ret += arg.Mangle();
    }
    return ret;
}

FunctionType::FunctionMangledName FunctionType::GetFunctionMangledName(bool appFunction,
                                                                       TemplateArgs *templateArgs) const {
    FunctionMangledName mangle = {};

    if (IsUnmangled()) {
        return mangle;
    }

    // Mangle internal functions name.
    if (!(IsExternC() || IsExternSYCL() || appFunction)) {
        mangle.suffix += mangleTemplateArgs(templateArgs);
        mangle.suffix += Mangle();
    }
    // Always add target suffix except extern "C" and extern "SYCL" internal cases.
    if (g->mangleFunctionsWithTarget) {
        if ((!appFunction && !IsExternC() && !IsExternSYCL()) || appFunction) {
            mangle.suffix += std::string("_") + g->target->GetISAString();
        }
    }
    // If the function is declared as regcall, add __regcall3__ prefix.
    if (IsRegCall()) {
        mangle.prefix += "__regcall3__";
    }
    return mangle;
}

std::vector<llvm::Type *> FunctionType::LLVMFunctionArgTypes(llvm::LLVMContext *ctx, bool removeMask) const {
    // Get the LLVM Type *s for the function arguments
    std::vector<llvm::Type *> llvmArgTypes;
    for (unsigned int i = 0; i < paramTypes.size(); ++i) {
        if (paramTypes[i] == nullptr) {
            Assert(m->errorCount > 0);
            return llvmArgTypes;
        }
        Assert(paramTypes[i]->IsVoidType() == false);

        const Type *argType = paramTypes[i];

        // We should cast pointers to generic address spaces
        // for ISPCExtenal functions (not-masked version) on Xe target
        llvm::Type *castedArgType = argType->LLVMType(ctx);

        if (IsISPCExternal() && removeMask) {
            if (argType->IsPointerType()) {
                const PointerType *argPtr = CastType<PointerType>(argType);
                if (argPtr->GetAddressSpace() == AddressSpace::ispc_default) {
                    argPtr = argPtr->GetWithAddrSpace(AddressSpace::ispc_generic);
                }
                castedArgType = argPtr->LLVMType(ctx);
            } else if (argType->IsReferenceType()) {
                const ReferenceType *refPtr = CastType<ReferenceType>(argType);
                if (refPtr->GetAddressSpace() == AddressSpace::ispc_default) {
                    refPtr = refPtr->GetWithAddrSpace(AddressSpace::ispc_generic);
                }
                castedArgType = refPtr->LLVMType(ctx);
            }
        }

        // For extern "SYCL" functions on Xe targets broadcast uniform parameters
        // to varying to match IGC signature by vISA level.
        if (g->target->isXeTarget() && IsExternSYCL()) {
            if (argType->IsUniformType()) {
                castedArgType = argType->GetAsVaryingType()->LLVMType(ctx);
            }
        }

        if (castedArgType == nullptr) {
            Assert(m->errorCount > 0);
            return llvmArgTypes;
        }
        llvmArgTypes.push_back(castedArgType);
    }

    // And add the function mask, if asked for
    if (!(removeMask || IsUnmasked() || IsISPCKernel())) {
        llvmArgTypes.push_back(LLVMTypes::MaskType);
    }
    return llvmArgTypes;
}

llvm::FunctionType *FunctionType::LLVMFunctionType(llvm::LLVMContext *ctx, bool removeMask) const {
    if (!g->target->isXeTarget() && IsTask() == true) {
        Assert(removeMask == false);
    }

    std::vector<llvm::Type *> llvmArgTypes = LLVMFunctionArgTypes(ctx, removeMask);

    std::vector<llvm::Type *> callTypes;
    if (IsTask() && (!g->target->isXeTarget())) {
        // Tasks take three arguments: a pointer to a struct that holds the
        // actual task arguments, the thread index, and the total number of
        // threads the tasks system has running.  (Task arguments are
        // marshalled in a struct so that it's easy to allocate space to
        // hold them until the task actually runs.)
        callTypes.push_back(llvm::PointerType::getUnqual(*ctx));
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

    if (returnType == nullptr) {
        Assert(m->errorCount > 0);
        return nullptr;
    }

    const Type *retType = returnType;

    llvm::Type *llvmReturnType = retType->LLVMType(g->ctx);
    // For extern "SYCL" functions on XE targets broadcast uniform return value
    // to varying to match IGC signature by vISA level.
    if (g->target->isXeTarget() && IsExternSYCL()) {
        if (!retType->IsVoidType() && retType->IsUniformType()) {
            llvmReturnType = retType->GetAsVaryingType()->LLVMType(ctx);
        }
    }
    if (llvmReturnType == nullptr) {
        return nullptr;
    }
    return llvm::FunctionType::get(llvmReturnType, callTypes, false);
}

llvm::Function *FunctionType::CreateLLVMFunction(const std::string &name, llvm::LLVMContext *ctx,
                                                 bool disableMask) const {
    llvm::FunctionType *llvmFunctionType = LLVMFunctionType(ctx, disableMask);
    if (llvmFunctionType == nullptr) {
        return nullptr;
    }
    // According to the LLVM philosophy, function declaration itself can't have internal linkage.
    // Function may have internal linkage only if it is defined in the module.
    // So here we set external linkage for all function declarations. It will be changed to internal
    // if the function is defined in the module in Function::GenerateIR().
    llvm::GlobalValue::LinkageTypes linkage = llvm::GlobalValue::ExternalLinkage;
    llvm::Function *function = llvm::Function::Create(llvmFunctionType, linkage, name.c_str(), m->module);
    return function;
}

unsigned int FunctionType::GetCallingConv() const {
    // Default calling convention on CPU targets is CallingConv::C.
    // If __vectorcall or __regcall is specified explicitly, corresponding
    // llvm::CallingConv will be used.
    // For Xe targets it is either CallingConv::SPIR_KERNEL for kernels or
    // CallingConv::SPIR_FUNC for all other functions.

    // If cdecl is provided, use it ignoring --vectorcall and xe target, it is
    // used inside stdlib to override this global parameters.
    if (IsCdecl()) {
        return (unsigned int)llvm::CallingConv::C;
    }

    if (IsRegCall() && (IsExternC() || IsExternSYCL())) {
        return (unsigned int)llvm::CallingConv::X86_RegCall;
    }

    if (g->target->isXeTarget()) {
        if (IsISPCKernel()) {
            return (unsigned int)llvm::CallingConv::SPIR_KERNEL;
        } else {
            return (unsigned int)llvm::CallingConv::SPIR_FUNC;
        }
    }

    if (g->calling_conv == CallingConv::x86_vectorcall) {
        if ((IsVectorCall() && IsExternC()) || !IsExternC()) {
            return (unsigned int)llvm::CallingConv::X86_VectorCall;
        }
    }
    return (unsigned int)llvm::CallingConv::C;
}

const std::string FunctionType::GetNameForCallConv() const {
    switch (GetCallingConv()) {
    case llvm::CallingConv::C:
        return "";
    case llvm::CallingConv::X86_VectorCall:
        return "__vectorcall";
    case llvm::CallingConv::X86_RegCall:
        return "__regcall";
    case llvm::CallingConv::SPIR_FUNC:
        return "spir_func";
    case llvm::CallingConv::SPIR_KERNEL:
        return "spir_kernel";
    default:
        return "<unknown>";
    }
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

///////////////////////////////////////////////////////////////////////////
// Type
// static methods?

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
            return nullptr;
        }
        return vt;
    } else {
        const AtomicType *at = CastType<AtomicType>(type);
        if (!at) {
            Error(pos,
                  "Non-atomic type \"%s\" can't be converted to vector type "
                  "for %s.",
                  type->GetString().c_str(), reason);
            return nullptr;
        }
        return new VectorType(at, vecSize);
    }
}

const Type *Type::MoreGeneralType(const Type *t0, const Type *t1, SourcePos pos, const char *reason, bool forceVarying,
                                  int vecSize) {
    Assert(reason != nullptr);

    // First, if one or both types are function types, convert them to
    // pointer to function types and then try again.
    if (CastType<FunctionType>(t0) || CastType<FunctionType>(t1)) {
        if (CastType<FunctionType>(t0)) {
            t0 = PointerType::GetUniform(t0);
        }
        if (CastType<FunctionType>(t1)) {
            t1 = PointerType::GetUniform(t1);
        }
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
        if (!t0 || !t1) {
            return nullptr;
        }
    }

    // Are they both the same type?  If so, we're done, QED.
    if (Type::Equal(t0, t1)) {
        return t0;
    }

    // If they're function types, it's hopeless if they didn't match in the
    // Type::Equal() call above.  Fail here so that we don't get into
    // trouble calling GetAsConstType()...
    if (CastType<FunctionType>(t0) || CastType<FunctionType>(t1)) {
        Error(pos, "Incompatible function types \"%s\" and \"%s\" in %s.", t0->GetString().c_str(),
              t1->GetString().c_str(), reason);
        return nullptr;
    }

    // Not the same types, but only a const/non-const difference?  Return
    // the non-const type as the more general one.
    if (Type::EqualIgnoringConst(t0, t1)) {
        return t0->GetAsNonConstType();
    }

    const PointerType *pt0 = CastType<PointerType>(t0);
    const PointerType *pt1 = CastType<PointerType>(t1);
    if (pt0 != nullptr && pt1 != nullptr) {
        if (PointerType::IsVoidPointer(pt0)) {
            return pt1;
        } else if (PointerType::IsVoidPointer(pt1)) {
            return pt0;
        } else {
            Error(pos,
                  "Conversion between incompatible pointer types \"%s\" "
                  "and \"%s\" isn't possible.",
                  t0->GetString().c_str(), t1->GetString().c_str());
            return nullptr;
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
            return nullptr;
        }
        const Type *t = MoreGeneralType(vt0->GetElementType(), vt1->GetElementType(), pos, reason, forceVarying);
        if (!t) {
            return nullptr;
        }

        // The 'more general' version of the two vector element types must
        // be an AtomicType (that's all that vectors can hold...)
        const AtomicType *at = CastType<AtomicType>(t);
        Assert(at != nullptr);

        return new VectorType(at, vt0->GetElementCount());
    } else if (vt0) {
        // If one type is a vector type but the other isn't, see if we can
        // promote the other one to a vector type.  This will fail and
        // return nullptr if t1 is e.g. an array type and it's illegal to have
        // a vector of it..
        const Type *t = MoreGeneralType(vt0->GetElementType(), t1, pos, reason, forceVarying);
        if (!t) {
            return nullptr;
        }

        const AtomicType *at = CastType<AtomicType>(t);
        Assert(at != nullptr);
        return new VectorType(at, vt0->GetElementCount());
    } else if (vt1) {
        // As in the above case, see if we can promote t0 to make a vector
        // that matches vt1.
        const Type *t = MoreGeneralType(t0, vt1->GetElementType(), pos, reason, forceVarying);
        if (!t) {
            return nullptr;
        }

        const AtomicType *at = CastType<AtomicType>(t);
        Assert(at != nullptr);
        return new VectorType(at, vt1->GetElementCount());
    }

    // TODO: what do we need to do about references here, if anything??

    const AtomicType *at0 = CastType<AtomicType>(t0->GetReferenceTarget());
    const AtomicType *at1 = CastType<AtomicType>(t1->GetReferenceTarget());

    const EnumType *et0 = CastType<EnumType>(t0->GetReferenceTarget());
    const EnumType *et1 = CastType<EnumType>(t1->GetReferenceTarget());
    if (et0 != nullptr && et1 != nullptr) {
        // Two different enum types -> make them uint32s...
        Assert(et0->IsVaryingType() == et1->IsVaryingType());
        return et0->IsVaryingType() ? AtomicType::VaryingUInt32 : AtomicType::UniformUInt32;
    } else if (et0 != nullptr) {
        if (at1 != nullptr) {
            // Enum type and atomic type -> convert the enum to the atomic type
            // TODO: should we return uint32 here, unless the atomic type is
            // a 64-bit atomic type, in which case we return that?
            return at1;
        } else {
            Error(pos,
                  "Implicit conversion from enum type \"%s\" to "
                  "non-atomic type \"%s\" for %s not possible.",
                  t0->GetString().c_str(), t1->GetString().c_str(), reason);
            return nullptr;
        }
    } else if (et1 != nullptr) {
        if (at0 != nullptr) {
            // Enum type and atomic type; see TODO above here as well...
            return at0;
        } else {
            Error(pos,
                  "Implicit conversion from enum type \"%s\" to "
                  "non-atomic type \"%s\" for %s not possible.",
                  t1->GetString().c_str(), t0->GetString().c_str(), reason);
            return nullptr;
        }
    }

    // Now all we can do is promote atomic types...
    if (at0 == nullptr || at1 == nullptr) {
        Assert(reason != nullptr);
        Error(pos, "Implicit conversion from type \"%s\" to \"%s\" for %s not possible.", t0->GetString().c_str(),
              t1->GetString().c_str(), reason);
        return nullptr;
    }

    // Finally, to determine which of the two atomic types is more general,
    // use the ordering of entries in the AtomicType::BasicType enumerator.
    return (int(at0->basicType) >= int(at1->basicType)) ? at0 : at1;
}

bool Type::IsBasicType(const Type *type) {
    return (CastType<AtomicType>(type) != nullptr || CastType<EnumType>(type) != nullptr ||
            CastType<PointerType>(type) != nullptr);
}

static bool lCheckTypeEquality(const Type *a, const Type *b, bool ignoreConst) {
    if (a == nullptr || b == nullptr) {
        return false;
    }

    if (ignoreConst == false && a->IsConstType() != b->IsConstType()) {
        return false;
    }

    const AtomicType *ata = CastType<AtomicType>(a);
    const AtomicType *atb = CastType<AtomicType>(b);
    if (ata != nullptr && atb != nullptr) {
        return ((ata->basicType == atb->basicType) && (ata->GetVariability() == atb->GetVariability()));
    }

    // For all of the other types, we need to see if we have the same two
    // general types.  If so, then we dig into the details of the type and
    // see if all of the relevant bits are equal...
    const EnumType *eta = CastType<EnumType>(a);
    const EnumType *etb = CastType<EnumType>(b);
    if (eta != nullptr && etb != nullptr) {
        // Kind of goofy, but this sufficies to check
        return (eta->GetSourcePos() == etb->GetSourcePos() && eta->GetVariability() == etb->GetVariability());
    }

    const ArrayType *arta = CastType<ArrayType>(a);
    const ArrayType *artb = CastType<ArrayType>(b);
    if (arta != nullptr && artb != nullptr) {
        return (arta->GetElementCount() == artb->GetElementCount() &&
                lCheckTypeEquality(arta->GetElementType(), artb->GetElementType(), ignoreConst));
    }

    const VectorType *vta = CastType<VectorType>(a);
    const VectorType *vtb = CastType<VectorType>(b);
    if (vta != nullptr && vtb != nullptr) {
        return (vta->GetElementCount() == vtb->GetElementCount() &&
                lCheckTypeEquality(vta->GetElementType(), vtb->GetElementType(), ignoreConst));
    }

    const StructType *sta = CastType<StructType>(a);
    const StructType *stb = CastType<StructType>(b);
    const UndefinedStructType *usta = CastType<UndefinedStructType>(a);
    const UndefinedStructType *ustb = CastType<UndefinedStructType>(b);
    if ((sta != nullptr || usta != nullptr) && (stb != nullptr || ustb != nullptr)) {
        // Report both defuned and undefined structs as equal if their
        // names are the same.
        if (a->GetVariability() != b->GetVariability()) {
            return false;
        }

        const std::string &namea = sta ? sta->GetStructName() : usta->GetStructName();
        const std::string &nameb = stb ? stb->GetStructName() : ustb->GetStructName();
        return (namea == nameb);
    }

    const PointerType *pta = CastType<PointerType>(a);
    const PointerType *ptb = CastType<PointerType>(b);
    if (pta != nullptr && ptb != nullptr) {
        return (pta->IsUniformType() == ptb->IsUniformType() && pta->IsSlice() == ptb->IsSlice() &&
                pta->IsFrozenSlice() == ptb->IsFrozenSlice() && pta->GetAddressSpace() == ptb->GetAddressSpace() &&
                lCheckTypeEquality(pta->GetBaseType(), ptb->GetBaseType(), ignoreConst));
    }

    const ReferenceType *rta = CastType<ReferenceType>(a);
    const ReferenceType *rtb = CastType<ReferenceType>(b);
    if (rta != nullptr && rtb != nullptr) {
        return (rta->GetAddressSpace() == rtb->GetAddressSpace() &&
                lCheckTypeEquality(rta->GetReferenceTarget(), rtb->GetReferenceTarget(), ignoreConst));
    }

    const FunctionType *fta = CastType<FunctionType>(a);
    const FunctionType *ftb = CastType<FunctionType>(b);
    if (fta != nullptr && ftb != nullptr) {
        // Both the return types and all of the argument types must match
        // for function types to match
        if (!lCheckTypeEquality(fta->GetReturnType(), ftb->GetReturnType(), ignoreConst)) {
            return false;
        }

        // TODO! should we compare all the flags properly? It looks like this
        // place was not updated when some flags were added
        if (fta->IsTask() != ftb->IsTask() || fta->IsExported() != ftb->IsExported() ||
            fta->IsExternC() != ftb->IsExternC() || fta->IsExternSYCL() != ftb->IsExternSYCL() ||
            fta->IsUnmasked() != ftb->IsUnmasked()) {
            return false;
        }

        if (fta->GetNumParameters() != ftb->GetNumParameters()) {
            return false;
        }

        for (int i = 0; i < fta->GetNumParameters(); ++i) {
            if (!lCheckTypeEquality(fta->GetParameterType(i), ftb->GetParameterType(i), ignoreConst)) {
                return false;
            }
        }

        return true;
    }

    const TemplateTypeParmType *ttpa = CastType<TemplateTypeParmType>(a);
    const TemplateTypeParmType *ttpb = CastType<TemplateTypeParmType>(b);
    if (ttpa != nullptr && ttpb != nullptr) {
        // Template type parameter types must have the same name to match.
        if (ttpa->GetName() != ttpb->GetName()) {
            return false;
        }
        // Variability should match, otherwise they are not equal.
        if (ttpa->GetVariability() != ttpb->GetVariability()) {
            return false;
        }
        return true;
    }

    return false;
}

bool Type::Equal(const Type *a, const Type *b) { return lCheckTypeEquality(a, b, false); }

bool Type::EqualIgnoringConst(const Type *a, const Type *b) { return lCheckTypeEquality(a, b, true); }
