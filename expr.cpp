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

/** @file expr.cpp
    @brief Implementations of expression classes
*/

#include "expr.h"
#include "type.h"
#include "sym.h"
#include "ctx.h"
#include "module.h"
#include "util.h"
#include "llvmutil.h"

#include <list>
#include <set>
#include <stdio.h>
#include <llvm/Module.h>
#include <llvm/Function.h>
#include <llvm/Type.h>
#include <llvm/DerivedTypes.h>
#include <llvm/LLVMContext.h>
#include <llvm/Instructions.h>
#include <llvm/CallingConv.h>
#include <llvm/Target/TargetData.h>
#include <llvm/Support/IRBuilder.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/Support/InstIterator.h>

/////////////////////////////////////////////////////////////////////////////////////
// Expr

llvm::Value *
Expr::GetLValue(FunctionEmitContext *ctx) const {
    // Expressions that can't provide an lvalue can just return NULL
    return NULL;
}


llvm::Constant *
Expr::GetConstant(const Type *type) const {
    // The default is failure; just return NULL
    return NULL;
}


Symbol *
Expr::GetBaseSymbol() const {
    // Not all expressions can do this, so provide a generally-useful
    // default
    return NULL;
}


/** If a conversion from 'fromAtomicType' to 'toAtomicType' may cause lost
    precision, issue a warning.  Don't warn for conversions to bool and
    conversions between signed and unsigned integers of the same size.
 */
static void
lMaybeIssuePrecisionWarning(const AtomicType *toAtomicType, 
                            const AtomicType *fromAtomicType, 
                            SourcePos pos, const char *errorMsgBase) {
    switch (toAtomicType->basicType) {
    case AtomicType::TYPE_BOOL:
    case AtomicType::TYPE_INT8:
    case AtomicType::TYPE_UINT8:
    case AtomicType::TYPE_INT16:
    case AtomicType::TYPE_UINT16:
    case AtomicType::TYPE_INT32:
    case AtomicType::TYPE_UINT32:
    case AtomicType::TYPE_FLOAT:
    case AtomicType::TYPE_INT64:
    case AtomicType::TYPE_UINT64:
    case AtomicType::TYPE_DOUBLE:
        if ((int)toAtomicType->basicType < (int)fromAtomicType->basicType &&
            toAtomicType->basicType != AtomicType::TYPE_BOOL &&
            !(toAtomicType->basicType == AtomicType::TYPE_INT8 && 
              fromAtomicType->basicType == AtomicType::TYPE_UINT8) &&
            !(toAtomicType->basicType == AtomicType::TYPE_INT16 && 
              fromAtomicType->basicType == AtomicType::TYPE_UINT16) &&
            !(toAtomicType->basicType == AtomicType::TYPE_INT32 && 
              fromAtomicType->basicType == AtomicType::TYPE_UINT32) &&
            !(toAtomicType->basicType == AtomicType::TYPE_INT64 && 
              fromAtomicType->basicType == AtomicType::TYPE_UINT64))
            Warning(pos, "Conversion from type \"%s\" to type \"%s\" for %s"
                    " may lose information.",
                    fromAtomicType->GetString().c_str(), toAtomicType->GetString().c_str(),
                    errorMsgBase);
        break;
    default:
        FATAL("logic error in lMaybeIssuePrecisionWarning()");
    }
}


Expr *
Expr::TypeConv(const Type *toType, const char *errorMsgBase, bool failureOk,
               bool issuePrecisionWarnings) {
    /* This function is way too long and complex.  Is type conversion stuff
       always this messy, or can this be cleaned up somehow? */
    assert(failureOk || errorMsgBase != NULL);

    const Type *fromType = GetType();
    if (toType == NULL || fromType == NULL)
        return this;

    // The types are equal; there's nothing to do
    if (Type::Equal(toType, fromType))
        return this;

    if (fromType == AtomicType::Void) {
        if (!failureOk)
            Error(pos, "Can't convert from \"void\" to \"%s\" for %s.",
                  toType->GetString().c_str(), errorMsgBase);
        return NULL;
    }

    if (toType == AtomicType::Void) {
        if (!failureOk)
            Error(pos, "Can't convert type \"%s\" to \"void\" for %s.",
                  fromType->GetString().c_str(), errorMsgBase);
        return NULL;
    }

    if (toType->IsUniformType() && fromType->IsVaryingType()) {
        if (!failureOk)
            Error(pos, "Can't convert from varying type \"%s\" to uniform "
                  "type \"%s\" for %s.", fromType->GetString().c_str(), 
                  toType->GetString().c_str(), errorMsgBase);
        return NULL;
    }

    // Convert from type T -> const T; just return a TypeCast expr, which
    // can handle this
    if (Type::Equal(toType, fromType->GetAsConstType()))
        return new TypeCastExpr(toType, this, pos);
    
    if (dynamic_cast<const ReferenceType *>(fromType)) {
        if (dynamic_cast<const ReferenceType *>(toType)) {
            // Convert from a reference to a type to a const reference to a type;
            // this is handled by TypeCastExpr
            if (Type::Equal(toType->GetReferenceTarget(),
                            fromType->GetReferenceTarget()->GetAsConstType()))
                return new TypeCastExpr(toType, this, pos);

            const ArrayType *atFrom = dynamic_cast<const ArrayType *>(fromType->GetReferenceTarget());
            const ArrayType *atTo = dynamic_cast<const ArrayType *>(toType->GetReferenceTarget());
            if (atFrom != NULL && atTo != NULL && 
                Type::Equal(atFrom->GetElementType(), atTo->GetElementType()))
                return new TypeCastExpr(toType, this, pos);

            else {
                if (!failureOk)
                    Error(pos, "Can't convert between incompatible reference types \"%s\" "
                          "and \"%s\" for %s.", fromType->GetString().c_str(),
                          toType->GetString().c_str(), errorMsgBase);
                return NULL;
            }
        }
        else {
            // convert from a reference T -> T
            Expr *fromExpr = new DereferenceExpr(this, pos);
            if (fromExpr->GetType() == NULL)
                return NULL;
            return fromExpr->TypeConv(toType, errorMsgBase, failureOk);
        }
    }
    else if (dynamic_cast<const ReferenceType *>(toType)) {
        // T -> reference T
        Expr *fromExpr = new ReferenceExpr(this, pos);
        if (fromExpr->GetType() == NULL)
            return NULL;
        return fromExpr->TypeConv(toType, errorMsgBase, failureOk);
    }
    else if (Type::Equal(toType, fromType->GetAsNonConstType()))
        // convert: const T -> T (as long as T isn't a reference)
        return new TypeCastExpr(toType, this, pos);

    fromType = fromType->GetReferenceTarget();
    toType = toType->GetReferenceTarget();
    // I don't think this is necessary
//CO    if (Type::Equal(toType, fromType))
//CO        return fromExpr;

    const ArrayType *toArrayType = dynamic_cast<const ArrayType *>(toType);
    const ArrayType *fromArrayType = dynamic_cast<const ArrayType *>(fromType);
    if (toArrayType && fromArrayType) {
        if (Type::Equal(toArrayType->GetElementType(), fromArrayType->GetElementType())) {
            // the case of different element counts should have returned
            // out earlier, yes??
            assert(toArrayType->GetElementCount() != fromArrayType->GetElementCount());
            return new TypeCastExpr(new ReferenceType(toType, false), this, pos);
        }
        else if (Type::Equal(toArrayType->GetElementType(), 
                             fromArrayType->GetElementType()->GetAsConstType())) {
            // T[x] -> const T[x]
            return new TypeCastExpr(new ReferenceType(toType, false), this, pos);
        }
        else {
            if (!failureOk)
                Error(pos, "Array type \"%s\" can't be converted to type \"%s\" for %s.",
                      fromType->GetString().c_str(), toType->GetString().c_str(),
                      errorMsgBase);
            return NULL;
        }
    }

    const VectorType *toVectorType = dynamic_cast<const VectorType *>(toType);
    const VectorType *fromVectorType = dynamic_cast<const VectorType *>(fromType);
    if (toVectorType && fromVectorType) {
        // converting e.g. int<n> -> float<n>
        if (fromVectorType->GetElementCount() != toVectorType->GetElementCount()) {
            if (!failureOk)
                Error(pos, "Can't convert between differently sized vector types "
                      "\"%s\" -> \"%s\" for %s.", fromType->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return NULL;
        }
        return new TypeCastExpr(toType, this, pos);
    }

    const StructType *toStructType = dynamic_cast<const StructType *>(toType);
    const StructType *fromStructType = dynamic_cast<const StructType *>(fromType);
    if (toStructType && fromStructType) {
        if (!Type::Equal(toStructType->GetAsUniformType()->GetAsConstType(),
                         fromStructType->GetAsUniformType()->GetAsConstType())) {
            if (!failureOk)
                Error(pos, "Can't convert between different struct types "
                      "\"%s\" -> \"%s\".", fromStructType->GetString().c_str(),
                      toStructType->GetString().c_str());
            return NULL;
        }

        return new TypeCastExpr(toType, this, pos);
    }

    const EnumType *toEnumType = dynamic_cast<const EnumType *>(toType);
    const EnumType *fromEnumType = dynamic_cast<const EnumType *>(fromType);
    if (toEnumType != NULL && fromEnumType != NULL) {
        // No implicit conversions between different enum types
        if (!Type::Equal(toEnumType->GetAsUniformType()->GetAsConstType(),
                         fromEnumType->GetAsUniformType()->GetAsConstType())) {
            if (!failureOk)
                Error(pos, "Can't convert between different enum types "
                      "\"%s\" -> \"%s\".", fromEnumType->GetString().c_str(),
                      toEnumType->GetString().c_str());
            return NULL;
        }

        return new TypeCastExpr(toType, this, pos);
    }

    const AtomicType *toAtomicType = dynamic_cast<const AtomicType *>(toType);
    const AtomicType *fromAtomicType = dynamic_cast<const AtomicType *>(fromType);

    // enum -> atomic (integer, generally...) is always ok
    if (fromEnumType != NULL) {
        assert(toAtomicType != NULL || toVectorType != NULL);
        return new TypeCastExpr(toType, this, pos);
    }

    // from here on out, the from type can only be atomic something or
    // other...
    if (fromAtomicType == NULL) {
        if (!failureOk)
            Error(pos, "Type conversion only possible from atomic types, not "
                  "from \"%s\" to \"%s\", for %s.", fromType->GetString().c_str(), 
                  toType->GetString().c_str(), errorMsgBase);
        return NULL;
    }

    // scalar -> short-vector conversions
    if (toVectorType != NULL)
        return new TypeCastExpr(toType, this, pos);

    // ok, it better be a scalar->scalar conversion of some sort by now
    if (toAtomicType == NULL) {
        if (!failureOk)
            Error(pos, "Type conversion only possible to atomic types, not "
                  "from \"%s\" to \"%s\", for %s.",
                  fromType->GetString().c_str(), toType->GetString().c_str(), 
                  errorMsgBase);
        return NULL;
    }

    if (!failureOk && issuePrecisionWarnings)
        lMaybeIssuePrecisionWarning(toAtomicType, fromAtomicType, pos, 
                                    errorMsgBase);

    return new TypeCastExpr(toType, this, pos);
}


///////////////////////////////////////////////////////////////////////////

/** Given an atomic or vector type, this returns a boolean type with the
    same "shape".  In other words, if the given type is a vector type of
    three uniform ints, the returned type is a vector type of three uniform
    bools. */
static const Type *
lMatchingBoolType(const Type *type) {
    bool uniformTest = type->IsUniformType();
    const AtomicType *boolBase = uniformTest ? AtomicType::UniformBool : 
                                               AtomicType::VaryingBool;
    const VectorType *vt = dynamic_cast<const VectorType *>(type);
    if (vt != NULL)
        return new VectorType(boolBase, vt->GetElementCount());
    else {
        assert(dynamic_cast<const AtomicType *>(type) != NULL);
        return boolBase;
    }
}

///////////////////////////////////////////////////////////////////////////
// UnaryExpr

static llvm::Constant *
lLLVMConstantValue(const Type *type, llvm::LLVMContext *ctx, double value) {
    const AtomicType *atomicType = dynamic_cast<const AtomicType *>(type);
    const EnumType *enumType = dynamic_cast<const EnumType *>(type);
    const VectorType *vectorType = dynamic_cast<const VectorType *>(type);

    // This function is only called with, and only works for atomic, enum,
    // and vector types.
    assert(atomicType != NULL || enumType != NULL || vectorType != NULL);

    if (atomicType != NULL || enumType != NULL) {
        // If it's an atomic or enuemrator type, then figure out which of
        // the llvmutil.h functions to call to get the corresponding
        // constant and then call it...
        bool isUniform = type->IsUniformType();
        AtomicType::BasicType basicType = (enumType != NULL) ? 
            AtomicType::TYPE_UINT32 : atomicType->basicType;

        switch (basicType) {
        case AtomicType::TYPE_VOID:
            FATAL("can't get constant value for void type");
            return NULL;
        case AtomicType::TYPE_BOOL:
            if (isUniform)
                return (value != 0.) ? LLVMTrue : LLVMFalse;
            else
                return LLVMBoolVector(value != 0.);
        case AtomicType::TYPE_INT8: {
            int i = (int)value;
            assert((double)i == value);
            return isUniform ? LLVMInt8(i) : LLVMInt8Vector(i);
        }
        case AtomicType::TYPE_UINT8: {
            unsigned int i = (unsigned int)value;
            return isUniform ? LLVMUInt8(i) : LLVMUInt8Vector(i);
        }
        case AtomicType::TYPE_INT16: {
            int i = (int)value;
            assert((double)i == value);
            return isUniform ? LLVMInt16(i) : LLVMInt16Vector(i);
        }
        case AtomicType::TYPE_UINT16: {
            unsigned int i = (unsigned int)value;
            return isUniform ? LLVMUInt16(i) : LLVMUInt16Vector(i);
        }
        case AtomicType::TYPE_INT32: {
            int i = (int)value;
            assert((double)i == value);
            return isUniform ? LLVMInt32(i) : LLVMInt32Vector(i);
        }
        case AtomicType::TYPE_UINT32: {
            unsigned int i = (unsigned int)value;
            return isUniform ? LLVMUInt32(i) : LLVMUInt32Vector(i);
        }
        case AtomicType::TYPE_FLOAT:
            return isUniform ? LLVMFloat((float)value) : 
                               LLVMFloatVector((float)value);
        case AtomicType::TYPE_UINT64: {
            uint64_t i = (uint64_t)value;
            assert(value == (int64_t)i);
            return isUniform ? LLVMUInt64(i) : LLVMUInt64Vector(i);
        }
        case AtomicType::TYPE_INT64: {
            int64_t i = (int64_t)value;
            assert((double)i == value);
            return isUniform ? LLVMInt64(i) : LLVMInt64Vector(i);
        }
        case AtomicType::TYPE_DOUBLE:
            return isUniform ? LLVMDouble(value) : LLVMDoubleVector(value);
        default:
            FATAL("logic error in lLLVMConstantValue");
            return NULL;
        }
    }

    // For vector types, first get the LLVM constant for the basetype with
    // a recursive call to lLLVMConstantValue().
    const Type *baseType = vectorType->GetBaseType();
    llvm::Constant *constElement = lLLVMConstantValue(baseType, ctx, value);
    LLVM_TYPE_CONST llvm::Type *llvmVectorType = vectorType->LLVMType(ctx);

    // Now create a constant version of the corresponding LLVM type that we
    // use to represent the VectorType.
    // FIXME: this is a little ugly in that the fact that ispc represents
    // uniform VectorTypes as LLVM VectorTypes and varying VectorTypes as
    // LLVM ArrayTypes leaks into the code here; it feels like this detail
    // should be better encapsulated?
    if (baseType->IsUniformType()) {
        LLVM_TYPE_CONST llvm::VectorType *lvt = 
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(llvmVectorType);
        assert(lvt != NULL);
        std::vector<llvm::Constant *> vals;
        for (unsigned int i = 0; i < lvt->getNumElements(); ++i)
            vals.push_back(constElement);
	return llvm::ConstantVector::get(vals);
    }
    else {
        LLVM_TYPE_CONST llvm::ArrayType *lat = 
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(llvmVectorType);
        assert(lat != NULL);
        std::vector<llvm::Constant *> vals;
        for (unsigned int i = 0; i < lat->getNumElements(); ++i)
            vals.push_back(constElement);
        return llvm::ConstantArray::get(lat, vals);
    }
}


/** Utility routine to emit code to do a {pre,post}-{inc,dec}rement of the
    given expresion.
 */
static llvm::Value *
lEmitPrePostIncDec(UnaryExpr::Op op, Expr *expr, SourcePos pos,
                   FunctionEmitContext *ctx) {
    const Type *type = expr->GetType();

    // Get both the lvalue and the rvalue of the given expression
    llvm::Value *lvalue = NULL, *rvalue = NULL;
    if (dynamic_cast<const ReferenceType *>(type) != NULL) {
        type = type->GetReferenceTarget();
        lvalue = expr->GetValue(ctx);

        Expr *deref = new DereferenceExpr(expr, expr->pos);
        rvalue = deref->GetValue(ctx);
    }
    else {
        lvalue = expr->GetLValue(ctx);
        rvalue = expr->GetValue(ctx);
    }

    if (lvalue == NULL) {
        // If we can't get a lvalue, then we have an error here 
        Error(expr->pos, "Can't %s-%s non-lvalues.",
              (op == UnaryExpr::PreInc || op == UnaryExpr::PreDec) ? "pre" : "post",
              (op == UnaryExpr::PreInc || op == UnaryExpr::PostInc) ? "increment" : "decrement");
        return NULL;
    }

    // Emit code to do the appropriate addition/subtraction to the
    // expression's old value
    ctx->SetDebugPos(pos);
    llvm::Value *binop = NULL;
    int delta = (op == UnaryExpr::PreInc || op == UnaryExpr::PostInc) ? 1 : -1;
    llvm::Constant *dval = lLLVMConstantValue(type, g->ctx, delta);
    if (!type->IsFloatType())
        binop = ctx->BinaryOperator(llvm::Instruction::Add, rvalue, 
                                    dval, "val_inc_or_dec");
    else
        binop = ctx->BinaryOperator(llvm::Instruction::FAdd, rvalue, 
                                    dval, "val_inc_or_dec");

#if 0
    if (type->IsUniformType()) {
        if (ctx->VaryingCFDepth() > 0)
            Warning(expr->pos, 
                    "Modifying \"uniform\" value under \"varying\" control flow.  Beware.");
    }
#endif

    // And store the result out to the lvalue
    ctx->StoreInst(binop, lvalue, ctx->GetMask(), type);

    // And then if it's a pre increment/decrement, return the final
    // computed result; otherwise return the previously-grabbed expression
    // value.
    return (op == UnaryExpr::PreInc || op == UnaryExpr::PreDec) ? binop : rvalue;
}



/** Utility routine to emit code to negate the given expression.
 */
static llvm::Value *
lEmitNegate(Expr *arg, SourcePos pos, FunctionEmitContext *ctx) {
    const Type *type = arg->GetType();
    llvm::Value *argVal = arg->GetValue(ctx);
    if (type == NULL || argVal == NULL)
        return NULL;

    // Negate by subtracting from zero...
    llvm::Value *zero = lLLVMConstantValue(type, g->ctx, 0.);
    ctx->SetDebugPos(pos);
    if (type->IsFloatType())
        return ctx->BinaryOperator(llvm::Instruction::FSub, zero, argVal, "fnegate");
    else {
        assert(type->IsIntType());
        return ctx->BinaryOperator(llvm::Instruction::Sub, zero, argVal, "inegate");
    }
}


UnaryExpr::UnaryExpr(Op o, Expr *e, SourcePos p) 
  : Expr(p), op(o) { 
    expr = e;
}


llvm::Value *
UnaryExpr::GetValue(FunctionEmitContext *ctx) const {
    if (expr == NULL)
        return NULL;

    ctx->SetDebugPos(pos);

    switch (op) {
    case PreInc:
    case PreDec:
    case PostInc:
    case PostDec:
        return lEmitPrePostIncDec(op, expr, pos, ctx);
    case Negate:
        return lEmitNegate(expr, pos, ctx);
    case LogicalNot: {
        llvm::Value *argVal = expr->GetValue(ctx);
        return ctx->NotOperator(argVal, "logicalnot");
    }
    case BitNot: {
        llvm::Value *argVal = expr->GetValue(ctx);
        return ctx->NotOperator(argVal, "bitnot");
    }
    default:
        FATAL("logic error");
        return NULL;
    }
}


const Type *
UnaryExpr::GetType() const {
    if (expr == NULL)
        return NULL;

    const Type *type = expr->GetType();
    if (type == NULL)
        return NULL;

    // For all unary expressions besides logical not, the returned type is
    // the same as the source type.  Logical not always returns a bool
    // type, with the same shape as the input type.
    switch (op) {
    case PreInc:
    case PreDec:
    case PostInc:
    case PostDec:
    case Negate:
    case BitNot: 
        return type;
    case LogicalNot:
        return lMatchingBoolType(type);
    default:
        FATAL("error");
        return NULL;
    }
}


Expr *
UnaryExpr::Optimize() {
    if (!expr)
        return NULL;

    expr = expr->Optimize();

    ConstExpr *constExpr = dynamic_cast<ConstExpr *>(expr);
    // If the operand isn't a constant, then we can't do any optimization
    // here...
    if (constExpr == NULL)
        return this;

    const Type *type = constExpr->GetType();
    bool isEnumType = dynamic_cast<const EnumType *>(type) != NULL;

    const Type *baseType = type->GetAsNonConstType()->GetAsUniformType();
    if (baseType == AtomicType::UniformInt8 ||
        baseType == AtomicType::UniformUInt8 ||
        baseType == AtomicType::UniformInt16 ||
        baseType == AtomicType::UniformUInt16 ||
        baseType == AtomicType::UniformInt64 ||
        baseType == AtomicType::UniformUInt64)
        // FIXME: should handle these at some point; for now we only do
        // constant folding for bool, int32 and float types...
        return this;

    switch (op) {
    case PreInc:
    case PreDec:
    case PostInc:
    case PostDec:
        // this shouldn't happen--it's illegal to modify a contant value..
        // An error will be issued elsewhere...
        return this;
    case Negate: {
        // Since we currently only handle int32 and floats here, it's safe
        // to stuff whatever we have into a double, do the negate as a
        // double, and then return a ConstExpr with the same type as the
        // original...
        double v[ISPC_MAX_NVEC];
        int count = constExpr->AsDouble(v);
        for (int i = 0; i < count; ++i)
            v[i] = -v[i];
        return new ConstExpr(constExpr, v);
    }
    case BitNot: {
        if (type == AtomicType::UniformInt32 || 
            type == AtomicType::VaryingInt32 ||
            type == AtomicType::UniformConstInt32 || 
            type == AtomicType::VaryingConstInt32) {
            int32_t v[ISPC_MAX_NVEC];
            int count = constExpr->AsInt32(v);
            for (int i = 0; i < count; ++i)
                v[i] = ~v[i];
            return new ConstExpr(type, v, pos);
        }
        else if (type == AtomicType::UniformUInt32 || 
                 type == AtomicType::VaryingUInt32 ||
                 type == AtomicType::UniformConstUInt32 || 
                 type == AtomicType::VaryingConstUInt32 ||
                 isEnumType == true) {
            uint32_t v[ISPC_MAX_NVEC];
            int count = constExpr->AsUInt32(v);
            for (int i = 0; i < count; ++i)
                v[i] = ~v[i];
            return new ConstExpr(type, v, pos);
        }
        else
            FATAL("unexpected type in UnaryExpr::Optimize() / BitNot case");
    }
    case LogicalNot: {
        assert(type == AtomicType::UniformBool || 
               type == AtomicType::VaryingBool ||
               type == AtomicType::UniformConstBool || 
               type == AtomicType::VaryingConstBool);
        bool v[ISPC_MAX_NVEC];
        int count = constExpr->AsBool(v);
        for (int i = 0; i < count; ++i)
            v[i] = !v[i];
        return new ConstExpr(type, v, pos);
    }
    default:
        FATAL("unexpected op in UnaryExpr::Optimize()");
        return NULL;
    }
}


Expr *
UnaryExpr::TypeCheck() {
    if (expr != NULL) 
        expr = expr->TypeCheck();
    if (expr == NULL)
        // something went wrong in type checking...
        return NULL;

    const Type *type = expr->GetType();
    if (type == NULL)
        return NULL;

    if (op == PreInc || op == PreDec || op == PostInc || op == PostDec) {
        if (!type->IsNumericType()) {
            Error(expr->pos, "Can only pre/post increment float and integer "
                  "types, not \"%s\".", type->GetString().c_str());
            return NULL;
        }
        return this;
    }

    // don't do this for pre/post increment/decrement
    if (dynamic_cast<const ReferenceType *>(type)) {
        expr = new DereferenceExpr(expr, pos);
        type = expr->GetType();
    }

    if (op == Negate) {
        if (!type->IsNumericType()) {
            Error(expr->pos, "Negate not allowed for non-numeric type \"%s\".", 
                  type->GetString().c_str());
            return NULL;
        }
    }
    else if (op == LogicalNot) {
        const Type *boolType = lMatchingBoolType(type);
        expr = expr->TypeConv(boolType, "logical not");
        if (!expr)
            return NULL;
    }
    else if (op == BitNot) {
        if (!type->IsIntType()) {
            Error(expr->pos, "~ operator can only be used with integer types, "
                  "not \"%s\".", type->GetString().c_str());
            return NULL;
        }
    }
    return this;
}


void
UnaryExpr::Print() const {
    if (!expr || !GetType())
        return;

    printf("[ %s ] (", GetType()->GetString().c_str());
    if (op == PreInc) printf("++");
    if (op == PreDec) printf("--");
    if (op == Negate) printf("-");
    if (op == LogicalNot) printf("!");
    if (op == BitNot) printf("~");
    printf("(");
    expr->Print();
    printf(")");
    if (op == PostInc) printf("++");
    if (op == PostDec) printf("--");
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// BinaryExpr

static const char *
lOpString(BinaryExpr::Op op) {
    switch (op) {
    case BinaryExpr::Add:        return "+";
    case BinaryExpr::Sub:        return "-";
    case BinaryExpr::Mul:        return "*";
    case BinaryExpr::Div:        return "/";
    case BinaryExpr::Mod:        return "%";
    case BinaryExpr::Shl:        return "<<";
    case BinaryExpr::Shr:        return ">>";
    case BinaryExpr::Lt:         return "<";
    case BinaryExpr::Gt:         return ">";
    case BinaryExpr::Le:         return "<=";
    case BinaryExpr::Ge:         return ">=";
    case BinaryExpr::Equal:      return "==";
    case BinaryExpr::NotEqual:   return "!=";
    case BinaryExpr::BitAnd:     return "&";
    case BinaryExpr::BitXor:     return "^";
    case BinaryExpr::BitOr:      return "|";
    case BinaryExpr::LogicalAnd: return "&&";
    case BinaryExpr::LogicalOr:  return "||";
    case BinaryExpr::Comma:      return ",";
    default:
        FATAL("unimplemented case in lOpString()");
        return "";
    }
}


/** Utility routine to emit the binary bitwise operator corresponding to
    the given BinaryExpr::Op. 
*/
static llvm::Value *
lEmitBinaryBitOp(BinaryExpr::Op op, llvm::Value *arg0Val,
                 llvm::Value *arg1Val, FunctionEmitContext *ctx) {
    llvm::Instruction::BinaryOps inst;
    switch (op) {
    case BinaryExpr::Shl:    inst = llvm::Instruction::Shl;  break;
    case BinaryExpr::Shr:    inst = llvm::Instruction::AShr; break; 
    case BinaryExpr::BitAnd: inst = llvm::Instruction::And;  break;
    case BinaryExpr::BitXor: inst = llvm::Instruction::Xor;  break;
    case BinaryExpr::BitOr:  inst = llvm::Instruction::Or;   break;
    default:
        FATAL("logic error in lEmitBinaryBitOp()");
        return NULL;
    }

    return ctx->BinaryOperator(inst, arg0Val, arg1Val, "bitop");
}


/** Utility routine to emit binary arithmetic operator based on the given
    BinaryExpr::Op.
*/
static llvm::Value *
lEmitBinaryArith(BinaryExpr::Op op, llvm::Value *e0Val, llvm::Value *e1Val,
                 const Type *type, FunctionEmitContext *ctx, SourcePos pos) {
    llvm::Instruction::BinaryOps inst;
    bool isFloatOp = type->IsFloatType();
    bool isUnsignedOp = type->IsUnsignedType();

    switch (op) {
    case BinaryExpr::Add:
        inst = isFloatOp ? llvm::Instruction::FAdd : llvm::Instruction::Add;
        break;
    case BinaryExpr::Sub:
        inst = isFloatOp ? llvm::Instruction::FSub : llvm::Instruction::Sub;
        break;
    case BinaryExpr::Mul:
        inst = isFloatOp ? llvm::Instruction::FMul : llvm::Instruction::Mul;
        break;
    case BinaryExpr::Div:
        if (type->IsVaryingType() && !isFloatOp)
            PerformanceWarning(pos, "Division with varying integer types is "
                               "very inefficient."); 
        inst = isFloatOp ? llvm::Instruction::FDiv : 
                (isUnsignedOp ? llvm::Instruction::UDiv : llvm::Instruction::SDiv);
        break;
    case BinaryExpr::Mod:
        if (type->IsVaryingType() && !isFloatOp)
            PerformanceWarning(pos, "Modulus operator with varying types is "
                               "very inefficient."); 
        inst = isFloatOp ? llvm::Instruction::FRem : 
                (isUnsignedOp ? llvm::Instruction::URem : llvm::Instruction::SRem);
        break;
    default:
        FATAL("Invalid op type passed to lEmitBinaryArith()");
        return NULL;
    }

    return ctx->BinaryOperator(inst, e0Val, e1Val, "binop");
}


/** Utility routine to emit a binary comparison operator based on the given
    BinaryExpr::Op.
 */
static llvm::Value *
lEmitBinaryCmp(BinaryExpr::Op op, llvm::Value *e0Val, llvm::Value *e1Val,
               const Type *type, FunctionEmitContext *ctx, SourcePos pos) {
    bool isFloatOp = type->IsFloatType();
    bool isUnsignedOp = type->IsUnsignedType();

    llvm::CmpInst::Predicate pred;
    switch (op) {
    case BinaryExpr::Lt:
        pred = isFloatOp ? llvm::CmpInst::FCMP_OLT : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT);
        break;
    case BinaryExpr::Gt:
        pred = isFloatOp ? llvm::CmpInst::FCMP_OGT : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT);
        break;
    case BinaryExpr::Le:
        pred = isFloatOp ? llvm::CmpInst::FCMP_OLE : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE);
        break;
    case BinaryExpr::Ge:
        pred = isFloatOp ? llvm::CmpInst::FCMP_OGE : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_UGE : llvm::CmpInst::ICMP_SGE);
        break;
    case BinaryExpr::Equal:
        pred = isFloatOp ? llvm::CmpInst::FCMP_OEQ : llvm::CmpInst::ICMP_EQ;
        break;
    case BinaryExpr::NotEqual:
        pred = isFloatOp ? llvm::CmpInst::FCMP_ONE : llvm::CmpInst::ICMP_NE;
        break;
    default:
        FATAL("error in lEmitBinaryCmp()");
        return NULL;
    }

    llvm::Value *cmp = ctx->CmpInst(isFloatOp ? llvm::Instruction::FCmp : 
                                    llvm::Instruction::ICmp,
                                    pred, e0Val, e1Val, "bincmp");
    // This is a little ugly: CmpInst returns i1 values, but we use vectors
    // of i32s for varying bool values; type convert the result here if
    // needed.
    if (type->IsVaryingType())
        cmp = ctx->I1VecToBoolVec(cmp);

    return cmp;
}


BinaryExpr::BinaryExpr(Op o, Expr *a, Expr *b, SourcePos p) 
    : Expr(p), op(o) {
    arg0 = a;
    arg1 = b;
}


llvm::Value *
BinaryExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!arg0 || !arg1)
        return NULL;

    llvm::Value *e0Val = arg0->GetValue(ctx);
    llvm::Value *e1Val = arg1->GetValue(ctx);
    ctx->SetDebugPos(pos);

    switch (op) {
    case Add:
    case Sub:
    case Mul:
    case Div:
    case Mod:
        return lEmitBinaryArith(op, e0Val, e1Val, arg0->GetType(), ctx, pos);
    case Lt:
    case Gt:
    case Le:
    case Ge:
    case Equal:
    case NotEqual:
        return lEmitBinaryCmp(op, e0Val, e1Val, arg0->GetType(), ctx, pos);
    case Shl:
    case Shr:
    case BitAnd:
    case BitXor:
    case BitOr: {
        if (op == Shr && arg1->GetType()->IsVaryingType() && 
            dynamic_cast<ConstExpr *>(arg1) == NULL)
            PerformanceWarning(pos, "Shift right is extremely inefficient for "
                               "varying shift amounts.");
        return lEmitBinaryBitOp(op, e0Val, e1Val, ctx);
    }
    case LogicalAnd:
        return ctx->BinaryOperator(llvm::Instruction::And, e0Val, e1Val,
                                   "logical_and");
    case LogicalOr:
        return ctx->BinaryOperator(llvm::Instruction::Or, e0Val, e1Val, 
                                   "logical_or");
    case Comma:
        return e1Val;
    default:
        FATAL("logic error");
        return NULL;
    }
}


const Type *
BinaryExpr::GetType() const {
    if (arg0 == NULL || arg1 == NULL)
        return NULL;

    const Type *type0 = arg0->GetType(), *type1 = arg1->GetType();
    if (type0 == NULL || type1 == NULL)
        return NULL;

    if (!type0->IsBoolType() && !type0->IsNumericType()) {
        Error(arg0->pos, "First operand to binary operator \"%s\" is of invalid "
              "type \"%s\".", lOpString(op), type0->GetString().c_str());
        return NULL;
    }
    if (!type1->IsBoolType() && !type1->IsNumericType()) {
        Error(arg1->pos,
              "Second operand to binary operator \"%s\" is of invalid "
              "type \"%s\".", lOpString(op), type1->GetString().c_str());
        return NULL;
    }

    const Type *promotedType = Type::MoreGeneralType(type0, type1, pos, 
                                                     lOpString(op));
    // I don't think that MoreGeneralType should be able to fail after the
    // type checks above.
    assert(promotedType != NULL);

    switch (op) {
    case Add:
    case Sub:
    case Mul:
    case Div:
    case Mod:
        return promotedType;
    case Lt:
    case Gt:
    case Le:
    case Ge:
    case Equal:
    case NotEqual:
    case LogicalAnd:
    case LogicalOr:
        return lMatchingBoolType(promotedType);
    case Shl:
    case Shr:
        return type1->IsVaryingType() ? type0->GetAsVaryingType() : type0;
    case BitAnd:
    case BitXor:
    case BitOr:
        return promotedType;
    case Comma:
        return arg1->GetType();
    default:
        FATAL("logic error in BinaryExpr::GetType()");
        return NULL;
    }
}


#define FOLD_OP(O, E)                           \
    case O:                                     \
        for (int i = 0; i < count; ++i)         \
            result[i] = (v0[i] E v1[i]);        \
        break

/** Constant fold the binary integer operations that aren't also applicable
    to floating-point types. 
*/
template <typename T> static ConstExpr *
lConstFoldBinIntOp(BinaryExpr::Op op, const T *v0, const T *v1, ConstExpr *carg0) {
    T result[ISPC_MAX_NVEC];
    int count = carg0->Count();
        
    switch (op) {
        FOLD_OP(BinaryExpr::Mod, %);
        FOLD_OP(BinaryExpr::Shl, <<);
        FOLD_OP(BinaryExpr::Shr, >>);
        FOLD_OP(BinaryExpr::BitAnd, &);
        FOLD_OP(BinaryExpr::BitXor, ^);
        FOLD_OP(BinaryExpr::BitOr, |);
    default:
        return NULL;
    }

    return new ConstExpr(carg0->GetType(), result, carg0->pos);
}


/** Constant fold the binary logical ops.
 */ 
template <typename T> static ConstExpr *
lConstFoldBinLogicalOp(BinaryExpr::Op op, const T *v0, const T *v1, ConstExpr *carg0) {
    bool result[ISPC_MAX_NVEC];
    int count = carg0->Count();

    switch (op) {
        FOLD_OP(BinaryExpr::Lt, <);
        FOLD_OP(BinaryExpr::Gt, >);
        FOLD_OP(BinaryExpr::Le, <=);
        FOLD_OP(BinaryExpr::Ge, >=);
        FOLD_OP(BinaryExpr::Equal, ==);
        FOLD_OP(BinaryExpr::NotEqual, !=);
        FOLD_OP(BinaryExpr::LogicalAnd, &&);
        FOLD_OP(BinaryExpr::LogicalOr, ||);
    default:
        return NULL;
    }

    const Type *rType = carg0->GetType()->IsUniformType() ? AtomicType::UniformBool : 
                                                        AtomicType::VaryingBool;
    return new ConstExpr(rType, result, carg0->pos);
}


/** Constant fold binary arithmetic ops.
 */
template <typename T> static ConstExpr *
lConstFoldBinArithOp(BinaryExpr::Op op, const T *v0, const T *v1, ConstExpr *carg0) {
    T result[ISPC_MAX_NVEC];
    int count = carg0->Count();

    switch (op) {
        FOLD_OP(BinaryExpr::Add, +);
        FOLD_OP(BinaryExpr::Sub, -);
        FOLD_OP(BinaryExpr::Mul, *);
        FOLD_OP(BinaryExpr::Div, /);
    default:
        return NULL;
    }
    
    return new ConstExpr(carg0->GetType(), result, carg0->pos);
}


/** Constant fold the various boolean binary ops.
 */
static ConstExpr *
lConstFoldBoolBinOp(BinaryExpr::Op op, const bool *v0, const bool *v1, 
                    ConstExpr *carg0) {
    bool result[ISPC_MAX_NVEC];
    int count = carg0->Count();

    switch (op) {
        FOLD_OP(BinaryExpr::BitAnd, &);
        FOLD_OP(BinaryExpr::BitXor, ^);
        FOLD_OP(BinaryExpr::BitOr, |);
        FOLD_OP(BinaryExpr::Lt, <);
        FOLD_OP(BinaryExpr::Gt, >);
        FOLD_OP(BinaryExpr::Le, <=);
        FOLD_OP(BinaryExpr::Ge, >=);
        FOLD_OP(BinaryExpr::Equal, ==);
        FOLD_OP(BinaryExpr::NotEqual, !=);
        FOLD_OP(BinaryExpr::LogicalAnd, &&);
        FOLD_OP(BinaryExpr::LogicalOr, ||);
    default:
        return NULL;
    }

    return new ConstExpr(carg0->GetType(), result, carg0->pos);
}


Expr *
BinaryExpr::Optimize() {
    if (arg0 != NULL) 
        arg0 = arg0->Optimize();
    if (arg1 != NULL) 
        arg1 = arg1->Optimize();

    if (!arg0 || !arg1)
        return NULL;

    ConstExpr *constArg0 = dynamic_cast<ConstExpr *>(arg0);
    ConstExpr *constArg1 = dynamic_cast<ConstExpr *>(arg1);

    if (g->opt.fastMath) {
        // optimizations related to division by floats..

        // transform x / const -> x * (1/const)
        if (op == Div && constArg1 != NULL) {
            const Type *type1 = constArg1->GetType();
            if (Type::Equal(type1, AtomicType::UniformFloat) ||
                Type::Equal(type1, AtomicType::VaryingFloat) ||
                Type::Equal(type1, AtomicType::UniformConstFloat) ||
                Type::Equal(type1, AtomicType::VaryingConstFloat)) {
                float inv[ISPC_MAX_NVEC];
                int count = constArg1->AsFloat(inv);
                for (int i = 0; i < count; ++i)
                    inv[i] = 1.f / inv[i];
                Expr *einv = new ConstExpr(type1, inv, constArg1->pos);
                Expr *e = new BinaryExpr(Mul, arg0, einv, pos);
                e = e->TypeCheck();
                if (e == NULL)
                    return NULL;
                return e->Optimize();
            }
        }

        // transform x / y -> x * rcp(y)
        if (op == Div) {
            const Type *type1 = arg1->GetType();
            if (Type::Equal(type1, AtomicType::UniformFloat) ||
                Type::Equal(type1, AtomicType::VaryingFloat) ||
                Type::Equal(type1, AtomicType::UniformConstFloat) ||
                Type::Equal(type1, AtomicType::VaryingConstFloat)) {
                // Get the symbol for the appropriate builtin
                std::vector<Symbol *> *rcpFuns = 
                    m->symbolTable->LookupFunction("rcp");
                if (rcpFuns != NULL) {
                    assert(rcpFuns->size() == 2);
                    Expr *rcpSymExpr = new FunctionSymbolExpr(rcpFuns, pos);
                    ExprList *args = new ExprList(arg1, arg1->pos);
                    Expr *rcpCall = new FunctionCallExpr(rcpSymExpr, args, 
                                                         arg1->pos, false);
                    rcpCall = rcpCall->TypeCheck();
                    if (rcpCall == NULL)
                        return NULL;
                    rcpCall = rcpCall->Optimize();
                    if (rcpCall == NULL)
                        return NULL;

                    Expr *ret = new BinaryExpr(Mul, arg0, rcpCall, pos);
                    ret = ret->TypeCheck();
                    if (ret == NULL)
                        return NULL;
                    return ret->Optimize();
                }
                else
                    Warning(pos, "rcp() not found from stdlib.  Can't apply "
                            "fast-math rcp optimization.");
            }
        }
    }

    // From here on out, we're just doing constant folding, so if both args
    // aren't constants then we're done...
    if (constArg0 == NULL || constArg1 == NULL)
        return this;

    assert(Type::Equal(arg0->GetType()->GetAsNonConstType(),
                       arg1->GetType()->GetAsNonConstType()));
    const Type *type = arg0->GetType()->GetAsNonConstType();
    if (type == AtomicType::UniformFloat || type == AtomicType::VaryingFloat) {
        float v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsFloat(v0);
        constArg1->AsFloat(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else 
            return this;
    }
    if (type == AtomicType::UniformDouble || type == AtomicType::VaryingDouble) {
        double v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsDouble(v0);
        constArg1->AsDouble(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else 
            return this;
    }
    if (type == AtomicType::UniformInt32 || type == AtomicType::VaryingInt32) {
        int32_t v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsInt32(v0);
        constArg1->AsInt32(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinIntOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else
            return this;
    }
    else if (type == AtomicType::UniformUInt32 || type == AtomicType::VaryingUInt32 ||
             dynamic_cast<const EnumType *>(type) != NULL) {
        uint32_t v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsUInt32(v0);
        constArg1->AsUInt32(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinIntOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else
            return this;
    }
    else if (type == AtomicType::UniformBool || type == AtomicType::VaryingBool) {
        bool v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsBool(v0);
        constArg1->AsBool(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBoolBinOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else 
            return this;
    }
    else
        return this;
}


Expr *
BinaryExpr::TypeCheck() {
    if (arg0 != NULL) 
        arg0 = arg0->TypeCheck();
    if (arg1 != NULL) 
        arg1 = arg1->TypeCheck();

    if (arg0 == NULL || arg1 == NULL)
        return NULL;

    const Type *type0 = arg0->GetType(), *type1 = arg1->GetType();
    if (type0 == NULL || type1 == NULL)
        return NULL;

    switch (op) {
    case Shl:
    case Shr:
    case BitAnd:
    case BitXor:
    case BitOr: {
        // Must have integer or bool-typed operands for these bit-related
        // ops; don't do any implicit conversions from floats here...
        if (!type0->IsIntType() && !type0->IsBoolType()) {
            Error(arg0->pos, "First operand to binary operator \"%s\" must be "
                  "an integer or bool.", lOpString(op));
            return NULL;
        }
        if (!type1->IsIntType() && !type1->IsBoolType()) {
            Error(arg1->pos, "Second operand to binary operator \"%s\" must be "
                  "an integer or bool.", lOpString(op));
            return NULL;
        }

        if (op == Shl || op == Shr) {
            bool isVarying = (type0->IsVaryingType() ||
                              type1->IsVaryingType());
            if (isVarying) {
                arg0 = arg0->TypeConv(type0->GetAsVaryingType(), "shift operator");
                type0 = arg0->GetType();
            }
            arg1 = arg1->TypeConv(type0, "shift operator", false, false);
            if (arg1 == NULL)
                return NULL;
        }
        else {
            const Type *promotedType = Type::MoreGeneralType(type0, type1, arg0->pos,
                                                             "binary bit op");
            if (promotedType == NULL)
                return NULL;

            arg0 = arg0->TypeConv(promotedType, "binary bit op");
            arg1 = arg1->TypeConv(promotedType, "binary bit op");
            if (arg0 == NULL || arg1 == NULL)
                return NULL;
        }
        return this;
    }
    case Add:
    case Sub:
    case Mul:
    case Div:
    case Mod:
    case Lt:
    case Gt:
    case Le:
    case Ge: {
        // Must be numeric type for these.  (And mod is special--can't be float)
        if (!type0->IsNumericType() || (op == Mod && type0->IsFloatType())) {
            Error(arg0->pos, "First operand to binary operator \"%s\" is of "
                  "invalid type \"%s\".", lOpString(op), 
                  type0->GetString().c_str());
            return NULL;
        }
        if (!type1->IsNumericType() || (op == Mod && type1->IsFloatType())) {
            Error(arg1->pos, "First operand to binary operator \"%s\" is of "
                  "invalid type \"%s\".", lOpString(op), 
                  type1->GetString().c_str());
            return NULL;
        }

        const Type *promotedType = Type::MoreGeneralType(type0, type1, arg0->pos,
                                                         lOpString(op));
        if (promotedType == NULL)
            return NULL;

        arg0 = arg0->TypeConv(promotedType, lOpString(op));
        arg1 = arg1->TypeConv(promotedType, lOpString(op));
        if (!arg0 || !arg1)
            return NULL;
        return this;
    }
    case Equal:
    case NotEqual: {
        if (!type0->IsBoolType() && !type0->IsNumericType()) {
            Error(arg0->pos,
                  "First operand to equality operator \"%s\" is of "
                  "non-comparable type \"%s\".", lOpString(op), 
                  type0->GetString().c_str());
            return NULL;
        }
        if (!type1->IsBoolType() && !type1->IsNumericType()) {
            Error(arg1->pos,
                  "Second operand to equality operator \"%s\" is of "
                  "non-comparable type \"%s\".", lOpString(op), 
                  type1->GetString().c_str());
            return NULL;
        }

        const Type *promotedType = 
            Type::MoreGeneralType(type0, type1, arg0->pos, lOpString(op));
        if (promotedType == NULL)
            return NULL;

        arg0 = arg0->TypeConv(promotedType, lOpString(op));
        arg1 = arg1->TypeConv(promotedType, lOpString(op));
        if (!arg0 || !arg1)
            return NULL;
        return this;
    }
    case LogicalAnd:
    case LogicalOr: {
        // We need to type convert to a boolean type of the more general
        // shape of the two types
        bool isUniform = (type0->IsUniformType() && type1->IsUniformType());
        const AtomicType *boolType = isUniform ? AtomicType::UniformBool : 
                                                 AtomicType::VaryingBool;
        const Type *destType = NULL;
        const VectorType *vtype0 = dynamic_cast<const VectorType *>(type0);
        const VectorType *vtype1 = dynamic_cast<const VectorType *>(type1);
        if (vtype0 && vtype1) {
            int sz0 = vtype0->GetElementCount(), sz1 = vtype1->GetElementCount();
            if (sz0 != sz1) {
                Error(pos, "Can't do logical operation \"%s\" between vector types of "
                      "different sizes (%d vs. %d).", lOpString(op), sz0, sz1);
                return NULL;
            }
            destType = new VectorType(boolType, sz0);
        }
        else if (vtype0)
            destType = new VectorType(boolType, vtype0->GetElementCount());
        else if (vtype1)
            destType = new VectorType(boolType, vtype1->GetElementCount());
        else
            destType = boolType;
            
        arg0 = arg0->TypeConv(destType, lOpString(op));
        arg1 = arg1->TypeConv(destType, lOpString(op));
        if (!arg0 || !arg1)
            return NULL;
        return this;
    }
    case Comma:
        return this;
    default:
        FATAL("logic error");
        return NULL;
    }
}


void
BinaryExpr::Print() const {
    if (!arg0 || !arg1 || !GetType())
        return;

    printf("[ %s ] (", GetType()->GetString().c_str());
    arg0->Print();
    printf(" %s ", lOpString(op));
    arg1->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// AssignExpr


/** Store the result of an assignment to the given location. 
 */
static void
lStoreAssignResult(llvm::Value *rv, llvm::Value *lv, const Type *type, 
                   FunctionEmitContext *ctx, Symbol *baseSym) {
    assert(baseSym->varyingCFDepth <= ctx->VaryingCFDepth());
    if (!g->opt.disableMaskedStoreToStore &&
        baseSym->varyingCFDepth == ctx->VaryingCFDepth() &&
        baseSym->isStatic == false &&
        dynamic_cast<const ReferenceType *>(baseSym->type) == NULL) {
        // If the variable is declared at the same varying control flow
        // depth as where it's being assigned, then we don't need to do any
        // masking but can just do the assignment as if all the lanes were
        // known to be on.  While this may lead to random/garbage values
        // written into the lanes that are off, by definition they will
        // never be accessed, since those lanes aren't executing, and won't
        // be executing at this scope or any other one before the variable
        // goes out of scope.
        ctx->StoreInst(rv, lv, LLVMMaskAllOn, type);
    }
    else
        ctx->StoreInst(rv, lv, ctx->GetMask(), type);
}


/** Emit code to do an "assignment + operation" operator, e.g. "+=".
 */
static llvm::Value *
lEmitOpAssign(AssignExpr::Op op, Expr *arg0, Expr *arg1, const Type *type, 
              Symbol *baseSym, SourcePos pos, FunctionEmitContext *ctx) {
    llvm::Value *lv = arg0->GetLValue(ctx);
    if (!lv) {
        // FIXME: I think this test is unnecessary and that this case
        // should be caught during typechecking
        Error(pos, "Can't assign to left-hand side of expression.");
        return NULL;
    }

    // Get the value on the right-hand side of the assignment+operation
    // operator and load the current value on the left-hand side.
    llvm::Value *rvalue = arg1->GetValue(ctx);
    ctx->SetDebugPos(pos);
    llvm::Value *oldLHS = ctx->LoadInst(lv, type, "opassign_load");

    // Map the operator to the corresponding BinaryExpr::Op operator
    BinaryExpr::Op basicop;
    switch (op) {
    case AssignExpr::MulAssign: basicop = BinaryExpr::Mul;    break;
    case AssignExpr::DivAssign: basicop = BinaryExpr::Div;    break;
    case AssignExpr::ModAssign: basicop = BinaryExpr::Mod;    break;
    case AssignExpr::AddAssign: basicop = BinaryExpr::Add;    break;
    case AssignExpr::SubAssign: basicop = BinaryExpr::Sub;    break;
    case AssignExpr::ShlAssign: basicop = BinaryExpr::Shl;    break;
    case AssignExpr::ShrAssign: basicop = BinaryExpr::Shr;    break;
    case AssignExpr::AndAssign: basicop = BinaryExpr::BitAnd; break;
    case AssignExpr::XorAssign: basicop = BinaryExpr::BitXor; break;
    case AssignExpr::OrAssign:  basicop = BinaryExpr::BitOr;  break;
    default:
        FATAL("logic error in lEmitOpAssign()");
        return NULL;
    }

    // Emit the code to compute the new value
    llvm::Value *newValue = NULL;
    switch (op) {
    case AssignExpr::MulAssign:
    case AssignExpr::DivAssign:
    case AssignExpr::ModAssign:
    case AssignExpr::AddAssign:
    case AssignExpr::SubAssign:
        newValue = lEmitBinaryArith(basicop, oldLHS, rvalue, type, ctx, pos);
        break;
    case AssignExpr::ShlAssign:
    case AssignExpr::ShrAssign:
    case AssignExpr::AndAssign:
    case AssignExpr::XorAssign:
    case AssignExpr::OrAssign:
        newValue = lEmitBinaryBitOp(basicop, oldLHS, rvalue, ctx);
        break;
    default:
        FATAL("logic error in lEmitOpAssign");
        return NULL;
    }

    // And store the result back to the lvalue.
    lStoreAssignResult(newValue, lv, type, ctx, baseSym);

    return newValue;
}


AssignExpr::AssignExpr(AssignExpr::Op o, Expr *a, Expr *b, SourcePos p) 
    : Expr(p), op(o) {
    lvalue = a;
    rvalue = b;
}


llvm::Value *
AssignExpr::GetValue(FunctionEmitContext *ctx) const {
    const Type *type = NULL;
    if (lvalue == NULL || rvalue == NULL || (type = GetType()) == NULL)
        return NULL;

    ctx->SetDebugPos(pos);

#if 0
    if (ctx->VaryingCFDepth() > 0 && type->IsUniformType())
        Warning(pos, "Modifying \"uniform\" value under \"varying\" control flow.  Beware.");
#endif

    Symbol *baseSym = lvalue->GetBaseSymbol();
    if (!baseSym) {
        // FIXME: I think that this check also is unnecessary and that this
        // case should be covered during type checking.
        Error(pos, "Left hand side of assignment statement can't be assigned to.");
        return NULL;
    }

    switch (op) {
    case Assign: {
        llvm::Value *lv = lvalue->GetLValue(ctx);
        if (!lv) {
            // FIXME: another, I believe, now unnecessary test?
            Error(lvalue->pos, "Can't assign to left-hand side of expression.");
            return NULL;
        }

        llvm::Value *rv = rvalue->GetValue(ctx);
        if (rv == NULL)
            return NULL;

        ctx->SetDebugPos(pos);

        // Warn if we're assigning a large array
        const ArrayType *at = dynamic_cast<const ArrayType *>(type);
        if (at && at->TotalElementCount() > 4)
            PerformanceWarning(pos, "Copying %d element array in assignment expression.",
                               at->TotalElementCount());

#if 0
        const StructType *st = dynamic_cast<const StructType *>(type);
        if (st != NULL) {
            bool anyUniform = false;
            for (int i = 0; i < st->NumElements(); ++i) {
                if (st->GetElementType(i)->IsUniformType())
                    anyUniform = true;
            }

            if (anyUniform && ctx->VaryingCFDepth() > 0)
                Warning(pos, "Modifying \"uniform\" value under \"varying\" "
                        "control flow.  Beware.");
        }
#endif

        lStoreAssignResult(rv, lv, type, ctx, baseSym);

        return rv;
    }
    case MulAssign:
    case DivAssign:
    case ModAssign:
    case AddAssign:
    case SubAssign:
    case ShlAssign:
    case ShrAssign:
    case AndAssign:
    case XorAssign:
    case OrAssign: {
        // This should be caught during type checking
        assert(!dynamic_cast<const ArrayType *>(type) &&
               !dynamic_cast<const StructType *>(type));
        return lEmitOpAssign(op, lvalue, rvalue, type, baseSym, pos, ctx);
    }
    default:
        FATAL("logic error in AssignExpr::GetValue()");
        return NULL;
    }
}


Expr *
AssignExpr::Optimize() {
    if (lvalue) 
        lvalue = lvalue->Optimize();
    if (rvalue) 
        rvalue = rvalue->Optimize();
    if (lvalue == NULL || rvalue == NULL)
        return NULL;

    return this;
}


const Type *
AssignExpr::GetType() const {
    return lvalue ? lvalue->GetType() : NULL;
}


Expr *
AssignExpr::TypeCheck() {
    bool lvalueIsReference = lvalue &&
        dynamic_cast<const ReferenceType *>(lvalue->GetType()) != NULL;
    bool rvalueIsReference = rvalue &&
        dynamic_cast<const ReferenceType *>(rvalue->GetType()) != NULL;

    // hack to allow asigning array references e.g. in a struct...
    if (lvalueIsReference &&
        !(rvalueIsReference && 
          dynamic_cast<const ArrayType *>(rvalue->GetType()->GetReferenceTarget())))
        lvalue = new DereferenceExpr(lvalue, lvalue->pos);

    if (lvalue != NULL) 
        lvalue = lvalue->TypeCheck();
    if (rvalue != NULL) 
        rvalue = rvalue->TypeCheck();
    if (rvalue != NULL && lvalue != NULL) 
        rvalue = rvalue->TypeConv(lvalue->GetType(), "assignment");
    if (rvalue == NULL || lvalue == NULL) 
        return NULL;

    if (lvalue->GetType()->IsConstType()) {
        Error(pos, "Can't assign to type \"%s\" on left-hand size of "
              "expression.", lvalue->GetType()->GetString().c_str());
        return NULL;
    }

    return this;
}


void
AssignExpr::Print() const {
    if (!lvalue || !rvalue || !GetType())
        return;

    printf("[%s] assign (", GetType()->GetString().c_str());
    lvalue->Print();
    printf(" ");
    if (op == Assign)    printf("=");
    if (op == MulAssign) printf("*=");
    if (op == DivAssign) printf("/=");
    if (op == ModAssign) printf("%%=");
    if (op == AddAssign) printf("+=");
    if (op == SubAssign) printf("-=");
    if (op == ShlAssign) printf("<<=");
    if (op == ShrAssign) printf(">>=");
    if (op == AndAssign) printf("&=");
    if (op == XorAssign) printf("^=");
    if (op == OrAssign)  printf("|=");
    printf(" ");
    rvalue->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// SelectExpr

SelectExpr::SelectExpr(Expr *t, Expr *e1, Expr *e2, SourcePos p) 
    : Expr(p) {
    test = t;
    expr1 = e1;
    expr2 = e2;
}


/** Emit code to select between two varying values based on a varying test
    value.
 */
static llvm::Value *
lEmitVaryingSelect(FunctionEmitContext *ctx, llvm::Value *test, 
                   llvm::Value *expr1, llvm::Value *expr2, 
                   const Type *type) {
    llvm::Value *resultPtr = ctx->AllocaInst(expr1->getType(), "selectexpr_tmp");
    // Don't need to worry about masking here
    ctx->StoreInst(expr2, resultPtr);
    // Use masking to conditionally store the expr1 values
    ctx->StoreInst(expr1, resultPtr, test, type);
    return ctx->LoadInst(resultPtr, type, "selectexpr_final");
}


llvm::Value *
SelectExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr1 || !expr2 || !test)
        return NULL;

    ctx->SetDebugPos(pos);

    const Type *testType = test->GetType()->GetAsNonConstType();
    // This should be taken care of during typechecking
    assert(testType->GetBaseType() == AtomicType::UniformBool ||
           testType->GetBaseType() == AtomicType::VaryingBool);

    const Type *type = expr1->GetType();
    // Type checking should also make sure this is the case
    assert(Type::Equal(type->GetAsNonConstType(), 
                       expr2->GetType()->GetAsNonConstType()));

    if (testType == AtomicType::UniformBool) {
        // Simple case of a single uniform bool test expression; we just
        // want one of the two expressions.  In this case, we can be
        // careful to evaluate just the one of the expressions that we need
        // the value of so that if the other one has side-effects or
        // accesses invalid memory, it doesn't execute.
        llvm::Value *testVal = test->GetValue(ctx);
        llvm::BasicBlock *testTrue = ctx->CreateBasicBlock("select_true");
        llvm::BasicBlock *testFalse = ctx->CreateBasicBlock("select_false");
        llvm::BasicBlock *testDone = ctx->CreateBasicBlock("select_done");
        ctx->BranchInst(testTrue, testFalse, testVal);

        ctx->SetCurrentBasicBlock(testTrue);
        llvm::Value *expr1Val = expr1->GetValue(ctx);
        // Note that truePred won't be necessarily equal to testTrue, in
        // case the expr1->GetValue() call changes the current basic block.
        llvm::BasicBlock *truePred = ctx->GetCurrentBasicBlock();
        ctx->BranchInst(testDone);

        ctx->SetCurrentBasicBlock(testFalse);
        llvm::Value *expr2Val = expr2->GetValue(ctx);
        // See comment above truePred for why we can't just assume we're in
        // the testFalse basic block here.
        llvm::BasicBlock *falsePred = ctx->GetCurrentBasicBlock();
        ctx->BranchInst(testDone);

        ctx->SetCurrentBasicBlock(testDone);
        llvm::PHINode *ret = ctx->PhiNode(expr1Val->getType(), 2, "select");
        ret->addIncoming(expr1Val, truePred);
        ret->addIncoming(expr2Val, falsePred);
        return ret;
    }
    else if (dynamic_cast<const VectorType *>(testType) == NULL) {
        // if the test is a varying bool type, then evaluate both of the
        // value expressions with the mask set appropriately and then do an
        // element-wise select to get the result
        llvm::Value *testVal = test->GetValue(ctx);
        assert(testVal->getType() == LLVMTypes::MaskType);
        llvm::Value *oldMask = ctx->GetMask();
        ctx->MaskAnd(oldMask, testVal);
        llvm::Value *expr1Val = expr1->GetValue(ctx);
        ctx->MaskAndNot(oldMask, testVal);
        llvm::Value *expr2Val = expr2->GetValue(ctx);
        ctx->SetMask(oldMask);

        return lEmitVaryingSelect(ctx, testVal, expr1Val, expr2Val, type);
    }
    else {
        // FIXME? Short-circuiting doesn't work in the case of
        // vector-valued test expressions.  (We could also just prohibit
        // these and place the issue in the user's hands...)
        llvm::Value *testVal = test->GetValue(ctx);
        llvm::Value *expr1Val = expr1->GetValue(ctx);
        llvm::Value *expr2Val = expr2->GetValue(ctx);

        ctx->SetDebugPos(pos);
        const VectorType *vt = dynamic_cast<const VectorType *>(type);
        // Things that typechecking should have caught
        assert(vt != NULL);
        assert(dynamic_cast<const VectorType *>(testType) != NULL &&
               (dynamic_cast<const VectorType *>(testType)->GetElementCount() == 
                vt->GetElementCount()));

        // Do an element-wise select  
        llvm::Value *result = llvm::UndefValue::get(type->LLVMType(g->ctx));
        for (int i = 0; i < vt->GetElementCount(); ++i) {
            llvm::Value *ti = ctx->ExtractInst(testVal, i, "");
            llvm::Value *e1i = ctx->ExtractInst(expr1Val, i, "");
            llvm::Value *e2i = ctx->ExtractInst(expr2Val, i, "");
            llvm::Value *sel = NULL;
            if (testType->IsUniformType())
                sel = ctx->SelectInst(ti, e1i, e2i);
            else
                sel = lEmitVaryingSelect(ctx, ti, e1i, e2i, vt->GetElementType());
            result = ctx->InsertInst(result, sel, i, "");
        }
        return result;
    }
}


const Type *
SelectExpr::GetType() const {
    if (!test || !expr1 || !expr2)
        return NULL;

    const Type *testType = test->GetType();
    const Type *expr1Type = expr1->GetType();
    const Type *expr2Type = expr2->GetType();

    if (!testType || !expr1Type || !expr2Type)
        return NULL;

    bool becomesVarying = (testType->IsVaryingType() || expr1Type->IsVaryingType() ||
                           expr2Type->IsVaryingType());
    // if expr1 and expr2 have different vector sizes, typechecking should fail...
    int testVecSize = dynamic_cast<const VectorType *>(testType) != NULL ?
        dynamic_cast<const VectorType *>(testType)->GetElementCount() : 0;
    int expr1VecSize = dynamic_cast<const VectorType *>(expr1Type) != NULL ?
        dynamic_cast<const VectorType *>(expr1Type)->GetElementCount() : 0;
//CO    int expr2VecSize = dynamic_cast<const VectorType *>(expr2Type) != NULL ?
//CO        dynamic_cast<const VectorType *>(expr2Type)->GetElementCount() : 0;
//CO    assert(testVecSize == expr1VecSize && expr1VecSize == expr2VecSize);
    // REMOVE? old test
    assert(!(testVecSize != 0 && expr1VecSize != 0 && testVecSize != expr1VecSize));
    
    int vectorSize = std::max(testVecSize, expr1VecSize);
    return Type::MoreGeneralType(expr1Type, expr2Type, pos, "select expression", 
                                 becomesVarying, vectorSize);
}


Expr *
SelectExpr::Optimize() {
    if (test) 
        test = test->Optimize();
    if (expr1) 
        expr1 = expr1->Optimize();
    if (expr2) 
        expr2 = expr2->Optimize();
    if (test == NULL || expr1 == NULL || expr2 == NULL)
        return NULL;

    return this;
}


Expr *
SelectExpr::TypeCheck() {
    if (test) 
        test = test->TypeCheck();
    if (expr1) 
        expr1 = expr1->TypeCheck();
    if (expr2) 
        expr2 = expr2->TypeCheck();

    if (test == NULL || expr1 == NULL || expr2 == NULL)
        return NULL;

    const Type *type1 = expr1->GetType(), *type2 = expr2->GetType();
    if (!type1 || !type2)
        return NULL;

    if (dynamic_cast<const ArrayType *>(type1)) {
        Error(pos, "Array type \"%s\" can't be used in select expression", 
              type1->GetString().c_str());
        return NULL;
    }
    if (dynamic_cast<const ArrayType *>(type2)) {
        Error(pos, "Array type \"%s\" can't be used in select expression", 
              type2->GetString().c_str());
        return NULL;
    }

    const Type *testType = test->GetType();
    if (testType == NULL)
        return NULL;
    test = test->TypeConv(lMatchingBoolType(testType), "select");
    if (testType == NULL)
        return NULL;
    testType = test->GetType();

    int testVecSize = dynamic_cast<const VectorType *>(testType) ?
        dynamic_cast<const VectorType *>(testType)->GetElementCount() : 0;
    const Type *promotedType = Type::MoreGeneralType(type1, type2, pos, "select expression", 
                                                     testType->IsVaryingType(), testVecSize);
    if (promotedType == NULL)
        return NULL;

    expr1 = expr1->TypeConv(promotedType, "select");
    expr2 = expr2->TypeConv(promotedType, "select");
    if (!expr1 || !expr2)
        return NULL;

    return this;
}


void
SelectExpr::Print() const {
    if (!test || !expr1 || !expr2 || !GetType())
        return;

    printf("[%s] (", GetType()->GetString().c_str());
    test->Print();
    printf(" ? ");
    expr1->Print();
    printf(" : ");
    expr2->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// FunctionCallExpr

static void
lPrintFunctionOverloads(const std::vector<Symbol *> &matches) {
    for (unsigned int i = 0; i < matches.size(); ++i) {
        const FunctionType *t = dynamic_cast<const FunctionType *>(matches[i]->type);
        assert(t != NULL);
        fprintf(stderr, "\t%s\n", t->GetString().c_str());
    }
}


static void
lPrintPassedTypes(const char *funName, const std::vector<Expr *> &argExprs) {
    fprintf(stderr, "Passed types:\n\t%s(", funName);
    for (unsigned int i = 0; i < argExprs.size(); ++i) {
        const Type *t;
        if (argExprs[i] != NULL && (t = argExprs[i]->GetType()) != NULL)
            fprintf(stderr, "%s%s", t->GetString().c_str(),
                    (i < argExprs.size()-1) ? ", " : ")\n\n");
        else
            fprintf(stderr, "(unknown type)%s", 
                    (i < argExprs.size()-1) ? ", " : ")\n\n");
    }
}

             
/** Helper function used for function overload resolution: returns true if
    the call argument's type exactly matches the function argument type
    (modulo a conversion to a const type if needed).
 */ 
static bool
lExactMatch(Expr *callArg, const Type *funcArgType) {
    const Type *callType = callArg->GetType();
// FIXME MOVE THESE TWO TO ALWAYS DO IT...
    if (dynamic_cast<const ReferenceType *>(callType) == NULL)
        callType = callType->GetAsNonConstType();
    if (dynamic_cast<const ReferenceType *>(funcArgType) != NULL && 
        dynamic_cast<const ReferenceType *>(callType) == NULL)
        callType = new ReferenceType(callType, funcArgType->IsConstType());

    return Type::Equal(callType, funcArgType);
}

/** Helper function used for function overload resolution: returns true if
    the call argument type and the function argument type match, modulo
    conversion to a reference type if needed.
 */
static bool
lMatchIgnoringReferences(Expr *callArg, const Type *funcArgType) {
    const Type *callType = callArg->GetType()->GetReferenceTarget();
    if (funcArgType->IsConstType())
        callType = callType->GetAsConstType();

    return Type::Equal(callType,
                       funcArgType->GetReferenceTarget());
}


/** Helper function used for function overload resolution: returns true if
    the call argument type and the function argument type match if we only
    do a uniform -> varying type conversion but otherwise have exactly the
    same type.
 */
static bool
lMatchIgnoringUniform(Expr *callArg, const Type *funcArgType) {
    const Type *callType = callArg->GetType();
    if (dynamic_cast<const ReferenceType *>(callType) == NULL)
        callType = callType->GetAsNonConstType();

    if (Type::Equal(callType, funcArgType))
        return true;

    return (callType->IsUniformType() && 
            funcArgType->IsVaryingType() &&
            Type::Equal(callType->GetAsVaryingType(), funcArgType));
}


/** Helper function used for function overload resolution: returns true if
    we can type convert from the call argument type to the function
    argument type, but without doing a uniform -> varying conversion.
 */
static bool
lMatchWithTypeConvSameVariability(Expr *callArg, const Type *funcArgType) {
    Expr *te = callArg->TypeConv(funcArgType, 
                                 "function call argument", true);
    return (te != NULL && 
            te->GetType()->IsUniformType() == callArg->GetType()->IsUniformType());
}


/** Helper function used for function overload resolution: returns true if
    there is any type conversion that gets us from the caller argument type
    to the function argument type.
 */
static bool
lMatchWithTypeConv(Expr *callArg, const Type *funcArgType) {
    Expr *te = callArg->TypeConv(funcArgType, 
                                 "function call argument", true);
    return (te != NULL);
}


/** See if we can find a single function from the set of overload options
    based on the predicate function passed in.  Returns true if no more
    tries should be made to find a match, either due to success from
    finding a single overloaded function that matches or failure due to
    finding multiple ambiguous matches.
 */
bool
FunctionCallExpr::tryResolve(bool (*matchFunc)(Expr *, const Type *)) {
    FunctionSymbolExpr *fse = dynamic_cast<FunctionSymbolExpr *>(func);
    if (!fse) 
        // error will be issued later if not calling an actual function
        return false;

    const char *funName = fse->candidateFunctions->front()->name.c_str();
    std::vector<Expr *> &callArgs = args->exprs;

    std::vector<Symbol *> matches;
    std::vector<Symbol *>::iterator iter;
    for (iter = fse->candidateFunctions->begin(); 
         iter != fse->candidateFunctions->end(); ++iter) {
        // Loop over the set of candidate functions and try each one
        Symbol *candidateFunction = *iter;
        const FunctionType *ft = 
            dynamic_cast<const FunctionType *>(candidateFunction->type);
        assert(ft != NULL);
        const std::vector<const Type *> &candArgTypes = ft->GetArgumentTypes();
        const std::vector<ConstExpr *> &argumentDefaults = ft->GetArgumentDefaults();

        // There's no way to match if the caller is passing more arguments
        // than this function instance takes.
        if (callArgs.size() > candArgTypes.size())
            continue;

        unsigned int i;
        // Note that we're looping over the caller arguments, not the
        // function arguments; it may be ok to have more arguments to the
        // function than are passed, if the function has default argument
        // values.  This case is handled below.
        for (i = 0; i < callArgs.size(); ++i) {
            // This may happen if there's an error earlier in compilation.
            // It's kind of a silly to redundantly discover this for each
            // potential match versus detecting this earlier in the
            // matching process and just giving up.
            if (!callArgs[i] || !callArgs[i]->GetType() || !candArgTypes[i] ||
                dynamic_cast<const FunctionType *>(callArgs[i]->GetType()) != NULL)
                return false;
            
            // See if this caller argument matches the type of the
            // corresponding function argument according to the given
            // predicate function.  If not, break out and stop trying.
            if (!matchFunc(callArgs[i], candArgTypes[i]))
                break;
        }
        if (i == callArgs.size()) {
            // All of the arguments matched!
            if (i == candArgTypes.size())
                // And we have exactly as many arguments as the function
                // wants, so we're done.
                matches.push_back(candidateFunction);
            else if (i < candArgTypes.size() && argumentDefaults[i] != NULL)
                // Otherwise we can still make it if there are default
                // arguments for the rest of the arguments!  Because in
                // Module::AddFunction() we have verified that once the
                // default arguments start, then all of the following ones
                // have them as well.  Therefore, we just need to check if
                // the arg we stopped at has a default value and we're
                // done.
                matches.push_back(candidateFunction);
            // otherwise, we don't have a match
        }
    }

    if (matches.size() == 0)
        return false;
    else if (matches.size() == 1) {
        fse->matchingFunc = matches[0];

        // fill in any function defaults required
        const FunctionType *ft = 
            dynamic_cast<const FunctionType *>(fse->matchingFunc->type);
        assert(ft != NULL);
        const std::vector<ConstExpr *> &argumentDefaults = ft->GetArgumentDefaults();
        const std::vector<const Type *> &argTypes = ft->GetArgumentTypes();
        assert(argumentDefaults.size() == argTypes.size());
        for (unsigned int i = callArgs.size(); i < argTypes.size(); ++i) {
            assert(argumentDefaults[i] != NULL);
            args->exprs.push_back(argumentDefaults[i]);
        }
        return true;
    }
    else {
        Error(fse->pos, "Multiple overloaded instances of function \"%s\" matched.",
              funName);
        lPrintFunctionOverloads(matches);
        lPrintPassedTypes(funName, args->exprs);
        // Stop trying to find more matches after failure
        return true;
    }
}


void
FunctionCallExpr::resolveFunctionOverloads() {
    FunctionSymbolExpr *fse = dynamic_cast<FunctionSymbolExpr *>(func);
    if (!fse) 
        // error will be issued later if not calling an actual function
        return;
    assert(args);
    
    // Try to find the best overload for the function...

    // Is there an exact match that doesn't require any argument type
    // conversion (other than converting type -> reference type)?
    if (tryResolve(lExactMatch))
        return;

    // Try to find a single match ignoring references
    if (tryResolve(lMatchIgnoringReferences))
        return;

    // TODO: next, try to find an exact match via type promotion--i.e. char
    // -> int, etc--things that don't lose data

    // Next try to see if there's a match via just uniform -> varying
    // promotions.  TODO: look for one with a minimal number of them?
    if (tryResolve(lMatchIgnoringUniform))
        return;

    // Try to find a match via type conversion, but don't change
    // unif->varying
    if (tryResolve(lMatchWithTypeConvSameVariability))
        return;
    
    // Last chance: try to find a match via arbitrary type conversion.
    if (tryResolve(lMatchWithTypeConv))
        return;

    // failure :-(
    const char *funName = fse->candidateFunctions->front()->name.c_str();
    Error(pos, "Unable to find matching overload for call to function \"%s\".",
          funName);
    fprintf(stderr, "Candidates are:\n");
    lPrintFunctionOverloads(*fse->candidateFunctions);
    lPrintPassedTypes(funName, args->exprs);
}


FunctionCallExpr::FunctionCallExpr(Expr *f, ExprList *a, SourcePos p, bool il) 
    : Expr(p) {
    func = f;
    args = a;
    isLaunch = il;

    resolveFunctionOverloads();
}


/** Starting from the function initialFunction, we're calling into
    calledFunc.  The question is: is this a recursive call back to
    initialFunc?  If it definitely is or if it may be, then return true.
    Return false if it definitely is not.
 */
static bool
lMayBeRecursiveCall(llvm::Function *calledFunc, 
                    llvm::Function *initialFunc,
                    std::set<llvm::Function *> &seenFuncs) {
    // Easy case: intrinsics aren't going to call functions themselves
    if (calledFunc->isIntrinsic())
        return false;

    std::string name = calledFunc->getName();
    if (name.size() > 2 && name[0] == '_' && name[1] == '_')
        // builtin stdlib function; none of these are recursive...
        return false;

    if (calledFunc->isDeclaration())
        // There's visibility into what the called function does without a
        // definition, so we have to be conservative
        return true;

    if (calledFunc == initialFunc)
        // hello recursive call
        return true;

    // Otherwise iterate over all of the instructions in the function.  If
    // any of them is a function call then check recursively..
    llvm::inst_iterator iter;
    for (iter = llvm::inst_begin(calledFunc); 
         iter != llvm::inst_end(calledFunc); ++iter) {
        llvm::Instruction *inst = &*iter;
        llvm::CallInst *ci = llvm::dyn_cast<llvm::CallInst>(inst);
        if (ci != NULL) {
            llvm::Function *nextCalledFunc = ci->getCalledFunction();
            // Don't repeatedly test functions we've seen before 
            if (seenFuncs.find(nextCalledFunc) == seenFuncs.end()) {
                seenFuncs.insert(nextCalledFunc);
                if (lMayBeRecursiveCall(nextCalledFunc, initialFunc, 
                                        seenFuncs))
                    return true;
            }
        }
    }
    return false;
}


llvm::Value *
FunctionCallExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!func || !args)
        return NULL;

    ctx->SetDebugPos(pos);

    FunctionSymbolExpr *fse = dynamic_cast<FunctionSymbolExpr *>(func);
    if (!fse) {
        Error(pos, "Invalid function name for function call.");
        return NULL;
    }

    if (!fse->matchingFunc) 
        // no overload match was found, get out of here..
        return NULL;

    Symbol *funSym = fse->matchingFunc;
    llvm::Function *callee = funSym->function;
    if (!callee) {
        Error(pos, "Symbol \"%s\" is not a function.", funSym->name.c_str());
        return NULL;
    }

    const FunctionType *ft = dynamic_cast<const FunctionType *>(funSym->type);
    assert(ft != NULL);
    bool isVoidFunc = (ft->GetReturnType() == AtomicType::Void);

    // Automatically convert function call args to references if needed.
    // FIXME: this should move to the TypeCheck() method... (but the
    // GetLValue call below needs a FunctionEmitContext, which is
    // problematic...)  
    std::vector<Expr *> callargs = args->exprs;
    const std::vector<const Type *> &argTypes = ft->GetArgumentTypes();
    bool err = false;
    for (unsigned int i = 0; i < callargs.size(); ++i) {
        Expr *argExpr = callargs[i];
        if (!argExpr)
            continue;

        // All arrays should already have been converted to reference types
        assert(dynamic_cast<const ArrayType *>(argTypes[i]) == NULL);

        if (dynamic_cast<const ReferenceType *>(argTypes[i])) {
            if (!dynamic_cast<const ReferenceType *>(argExpr->GetType())) {
                // The function wants a reference type but the argument
                // being passed isn't already a reference.
                if (argExpr->GetLValue(ctx) == NULL) {
                    // If it doesn't have an lvalue, then we can't make it
                    // a reference, so issue an error.
                    // FIXME: for const reference parameters, we could
                    // store the expr's value to alloca'ed memory and then
                    // pass a reference to that...
                    Error(pos, "Can't pass non-lvalue as \"reference\" parameter \"%s\" "
                          "to function \"%s\".", ft->GetArgumentName(i).c_str(), 
                          funSym->name.c_str());
                    err = true;
                }
                else
                    argExpr = new ReferenceExpr(argExpr, argExpr->pos);
            }
        }

        // Do whatever type conversion is needed
        argExpr = argExpr->TypeConv(argTypes[i], "function call argument");
        // The function overload resolution code should have ensured that
        // we can successfully do any type conversions needed here.
        assert(argExpr != NULL);
        callargs[i] = argExpr;
    }
    if (err)
        return NULL;

    // Now evaluate the values of all of the parameters being passed.  We
    // need to evaluate these first here, since their GetValue() calls may
    // change the current basic block (e.g. if one of these is itself a
    // function call expr...); we need to basic blocks to stay consistent
    // below when we emit the code that does the actual funciton call.
    std::vector<llvm::Value *> argVals;
    std::vector<llvm::Value *> storedArgValPtrs, argValLValues;
    for (unsigned int i = 0; i < callargs.size(); ++i) {
        Expr *argExpr = callargs[i];
        if (!argExpr)
            // give up; we hit an error earlier
            return NULL;

        llvm::Value *argValue = argExpr->GetValue(ctx);
        if (!argValue)
            // something went wrong in evaluating the argument's
            // expression, so give up on this
            return NULL;

        if (dynamic_cast<const ReferenceType *>(argTypes[i]) &&
            !llvm::isa<const llvm::PointerType>(argValue->getType())) {
            assert(llvm::isa<const llvm::ArrayType>(argValue->getType()));
            // if the parameter is a reference and the lvalue needs a
            // gather to pull it together, then do the gather here and
            // store the result to local memory, so that we can pass the
            // single pointer to the local memory that is needed for the
            // reference.  Below, we'll copy the result back to the varying
            // lvalue pointer we have here.  (== pass by value/result)
            const ReferenceType *rt = 
                dynamic_cast<const ReferenceType *>(argExpr->GetType());
            assert(rt != NULL);
            const Type *type = rt->GetReferenceTarget();

            llvm::Value *ptr = ctx->AllocaInst(type->LLVMType(g->ctx), "arg");
            llvm::Value *val = ctx->LoadInst(argValue, type);
            ctx->StoreInst(val, ptr);
            storedArgValPtrs.push_back(ptr);
            argValLValues.push_back(argValue);
            argVals.push_back(ptr);
        }
        else {
            argVals.push_back(argValue);
            storedArgValPtrs.push_back(NULL);
            argValLValues.push_back(NULL);
        }
    }

    // We sometimes need to check to see if the mask is all off here;
    // specifically, if the mask is all off and we call a recursive
    // function, then we will probably have an unsesirable infinite loop.
    ctx->SetDebugPos(pos);
    llvm::BasicBlock *bDoCall = ctx->CreateBasicBlock("funcall_mask_ok");
    llvm::BasicBlock *bSkip = ctx->CreateBasicBlock("funcall_mask_off");
    llvm::BasicBlock *bAfter = ctx->CreateBasicBlock("after_funcall");
    llvm::Function *currentFunc = ctx->GetCurrentBasicBlock()->getParent();

    // If we need to check the mask (it may be a recursive call, possibly
    // transitively), or we're launching a task, which is expensive and
    // thus probably always worth checking, then use the mask to choose
    // whether to go to the bDoCallBlock or the bSkip block
    std::set<llvm::Function *> seenFuncs;
    seenFuncs.insert(currentFunc);
    if (ft->isTask || lMayBeRecursiveCall(callee, currentFunc, seenFuncs)) {
        Debug(pos, "Checking mask before function call \"%s\".", funSym->name.c_str());
        ctx->BranchIfMaskAny(bDoCall, bSkip);
    }
    else
        // If we don't need to check the mask, then always to the call;
        // just jump to bDoCall
        ctx->BranchInst(bDoCall);
    
    // And the bSkip block just jumps immediately to bAfter.  So why do we
    // need it?  So the phi node below can easily tell what paths are
    // going into it
    ctx->SetCurrentBasicBlock(bSkip);
    ctx->BranchInst(bAfter);

    // Emit the code to do the function call
    ctx->SetCurrentBasicBlock(bDoCall);

    llvm::Value *retVal = NULL;
    ctx->SetDebugPos(pos);
    if (ft->isTask)
        ctx->LaunchInst(callee, argVals);
    else {
        // Most of the time, the mask is passed as the last argument.  this
        // isn't the case for things like SSE intrinsics and extern "C"
        // functions from the application.
        assert(callargs.size() + 1 == callee->arg_size() ||
               callargs.size() == callee->arg_size());

        if (callargs.size() + 1 == callee->arg_size())
            argVals.push_back(ctx->GetMask());

        retVal = ctx->CallInst(callee, argVals, isVoidFunc ? "" : "calltmp");
    }

    // For anything we had to do as pass by value/result, copy the
    // corresponding reference values back out
    for (unsigned int i = 0; i < storedArgValPtrs.size(); ++i) {
        llvm::Value *ptr = storedArgValPtrs[i];
        if (ptr != NULL) {
            const ReferenceType *rt = 
                dynamic_cast<const ReferenceType *>(callargs[i]->GetType());
            assert(rt != NULL);
            llvm::Value *load = ctx->LoadInst(ptr, rt->GetReferenceTarget(),
                                              "load_ref");
            // FIXME: apply the "don't do blending" optimization here if
            // appropriate?
            ctx->StoreInst(load, argValLValues[i], ctx->GetMask(), 
                           rt->GetReferenceTarget());
        }
    }

    // And jump out to the 'after funciton call' basic block
    ctx->BranchInst(bAfter);
    ctx->SetCurrentBasicBlock(bAfter);

    if (isVoidFunc)
        return NULL;

    // The return value for the non-void case is either undefined or the
    // function return value, depending on whether we actually ran the code
    // path that called the function or not.
    LLVM_TYPE_CONST llvm::Type *lrType = ft->GetReturnType()->LLVMType(g->ctx);
    llvm::PHINode *ret = ctx->PhiNode(lrType, 2, "fun_ret");
    assert(retVal != NULL);
    ret->addIncoming(llvm::UndefValue::get(lrType), bSkip);
    ret->addIncoming(retVal, bDoCall);
    return ret;
}


const Type *
FunctionCallExpr::GetType() const {
    FunctionSymbolExpr *fse = dynamic_cast<FunctionSymbolExpr *>(func);
    if (fse && fse->matchingFunc) {
        const FunctionType *ft = 
            dynamic_cast<const FunctionType *>(fse->matchingFunc->type);
        assert(ft != NULL);
        return ft->GetReturnType();
    }
    else
        return NULL;
}


Expr *
FunctionCallExpr::Optimize() {
    if (func) 
        func = func->Optimize();
    if (args) 
        args = args->Optimize();
    if (!func || !args)
        return NULL;
        
    return this;
}


Expr *
FunctionCallExpr::TypeCheck() {
    if (func) {
        func = func->TypeCheck();
        if (func != NULL) {
            const FunctionType *ft = dynamic_cast<const FunctionType *>(func->GetType());
            if (ft != NULL) {
                if (ft->isTask) {
                    if (!isLaunch)
                        Error(pos, "\"launch\" expression needed to call function "
                              "with \"task\" qualifier.");
                }
                else if (isLaunch)
                    Error(pos, "\"launch\" expression illegal with non-\"task\"-"
                          "qualified function.");
            }
            else
                Error(pos, "Valid function name must be used for function call.");
        }
    }

    if (args) 
        args = args->TypeCheck();

    if (!func || !args)
        return NULL;
    return this;
}


void
FunctionCallExpr::Print() const {
    if (!func || !args || !GetType())
        return;

    printf("[%s] funcall %s ", GetType()->GetString().c_str(),
           isLaunch ? "launch" : "");
    func->Print();
    printf(" args (");
    args->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// ExprList

llvm::Value *
ExprList::GetValue(FunctionEmitContext *ctx) const {
    FATAL("ExprList::GetValue() should never be called");
    return NULL;
}


const Type *
ExprList::GetType() const {
    FATAL("ExprList::GetType() should never be called");
    return NULL;
}


ExprList *
ExprList::Optimize() {
    for (unsigned int i = 0; i < exprs.size(); ++i)
        if (exprs[i])
            exprs[i] = exprs[i]->Optimize();
    return this;
}


ExprList *
ExprList::TypeCheck() {
    for (unsigned int i = 0; i < exprs.size(); ++i)
        if (exprs[i])
            exprs[i] = exprs[i]->TypeCheck();
    return this;
}


llvm::Constant *
ExprList::GetConstant(const Type *type) const {
    const CollectionType *collectionType = 
        dynamic_cast<const CollectionType *>(type);
    if (collectionType == NULL)
        return NULL;

    std::string name;
    if (dynamic_cast<const StructType *>(type) != NULL)
        name = "struct";
    else if (dynamic_cast<const ArrayType *>(type) != NULL) 
        name = "array";
    else if (dynamic_cast<const VectorType *>(type) != NULL) 
        name = "vector";
    else 
        FATAL("Unexpected CollectionType in ExprList::GetConstant()");

    if ((int)exprs.size() != collectionType->GetElementCount()) {
        Error(pos, "Initializer list for %s \"%s\" must have %d elements "
              "(has %d).", name.c_str(), collectionType->GetString().c_str(),
              collectionType->GetElementCount(), (int)exprs.size());
        return NULL;
    }

    std::vector<llvm::Constant *> cv;
    for (unsigned int i = 0; i < exprs.size(); ++i) {
        if (exprs[i] == NULL)
            return NULL;
        const Type *elementType = collectionType->GetElementType(i);
        llvm::Constant *c = exprs[i]->GetConstant(elementType);
        if (c == NULL)
            // If this list element couldn't convert to the right constant
            // type for the corresponding collection member, then give up.
            return NULL;
        cv.push_back(c);
    }

    if (dynamic_cast<const StructType *>(type) != NULL) {
#if defined(LLVM_2_8) || defined(LLVM_2_9)
        return llvm::ConstantStruct::get(*g->ctx, cv, false);
#else
        LLVM_TYPE_CONST llvm::StructType *llvmStructType =
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::StructType>(collectionType->LLVMType(g->ctx));
        assert(llvmStructType != NULL);
        return llvm::ConstantStruct::get(llvmStructType, cv);
#endif
    }
    else {
        LLVM_TYPE_CONST llvm::Type *lt = type->LLVMType(g->ctx);
        LLVM_TYPE_CONST llvm::ArrayType *lat = 
            llvm::dyn_cast<LLVM_TYPE_CONST llvm::ArrayType>(lt);
        // FIXME: should the assert below validly fail for uniform vectors
        // now?  Need a test case to reproduce it and then to be sure we
        // have the right fix; leave the assert until we can hit it...
        assert(lat != NULL);
        return llvm::ConstantArray::get(lat, cv);
    }
    return NULL;
}


void
ExprList::Print() const {
    printf("expr list (");
    for (unsigned int i = 0; i < exprs.size(); ++i) {
        if (exprs[i] != NULL)
            exprs[i]->Print();
        printf("%s", (i == exprs.size() - 1) ? ")" : ", ");
    }
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// IndexExpr

IndexExpr::IndexExpr(Expr *a, Expr *i, SourcePos p) 
    : Expr(p) {
    arrayOrVector = a;
    index = i;
}


// FIXME: This is an ugly hack--if we're indexing into a uniform ispc
// VectorType, then this bitcasts the corresponding llvm::VectorType value
// to be a pointer to the vector's element type, so that a GEP to index
// from the pointer indices elements of the llvm::VectorType and doesn't
// incorrectly try to index into an array of llvm::VectorType instances.

static llvm::Value *
lCastUniformVectorBasePtr(llvm::Value *ptr, FunctionEmitContext *ctx) {
    LLVM_TYPE_CONST llvm::PointerType *baseType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::PointerType>(ptr->getType());
    if (!baseType)
        return ptr;

    LLVM_TYPE_CONST llvm::VectorType *baseEltVecType = 
        llvm::dyn_cast<LLVM_TYPE_CONST llvm::VectorType>(baseType->getElementType());
    if (!baseEltVecType)
        return ptr;

    LLVM_TYPE_CONST llvm::Type *vecEltType = baseEltVecType->getElementType();
    int numElts = baseEltVecType->getNumElements();
    LLVM_TYPE_CONST llvm::Type *castType = 
        llvm::PointerType::get(llvm::ArrayType::get(vecEltType, numElts), 0);
    return ctx->BitCastInst(ptr, castType);
}


llvm::Value *
IndexExpr::GetValue(FunctionEmitContext *ctx) const {
    const Type *arrayOrVectorType;
    if (arrayOrVector == NULL || index == NULL || 
        ((arrayOrVectorType = arrayOrVector->GetType()) == NULL))
        return NULL;

    ctx->SetDebugPos(pos);
    llvm::Value *lvalue = GetLValue(ctx);
    if (!lvalue) {
        // We may be indexing into a temporary that hasn't hit memory, so
        // get the full value and stuff it into temporary alloca'd space so
        // that we can index from there...
        llvm::Value *val = arrayOrVector->GetValue(ctx);
        if (val == NULL) {
            assert(m->errorCount > 0);
            return NULL;
        }
        ctx->SetDebugPos(pos);
        llvm::Value *ptr = ctx->AllocaInst(arrayOrVectorType->LLVMType(g->ctx), 
                                           "array_tmp");
        ctx->StoreInst(val, ptr);
        ptr = lCastUniformVectorBasePtr(ptr, ctx);
        lvalue = ctx->GetElementPtrInst(ptr, LLVMInt32(0), index->GetValue(ctx));
    }

    ctx->SetDebugPos(pos);
    return ctx->LoadInst(lvalue, GetType(), "index");
}


const Type *
IndexExpr::GetType() const {
    const Type *arrayOrVectorType, *indexType;
    if (!arrayOrVector || !index || 
        ((arrayOrVectorType = arrayOrVector->GetType()) == NULL) ||
        ((indexType = index->GetType()) == NULL))
        return NULL;

    const SequentialType *sequentialType = 
        dynamic_cast<const SequentialType *>(arrayOrVectorType->GetReferenceTarget());
    // Typechecking should have caught this...
    assert(sequentialType != NULL);

    const Type *elementType = sequentialType->GetElementType();
    if (indexType->IsUniformType())
        // If the index is uniform, the resulting type is just whatever the
        // element type is
        return elementType;
    else
        // A varying index into uniform array/vector -> varying type (and
        // same for varying array of course...)
        return elementType->GetAsVaryingType();
}


Symbol *
IndexExpr::GetBaseSymbol() const {
    return arrayOrVector ? arrayOrVector->GetBaseSymbol() : NULL;
}


llvm::Value *
IndexExpr::GetLValue(FunctionEmitContext *ctx) const {
    const Type *type;
    if (!arrayOrVector || !index || ((type = arrayOrVector->GetType()) == NULL))
        return NULL;

    ctx->SetDebugPos(pos);
    llvm::Value *basePtr = NULL;
    if (dynamic_cast<const ArrayType *>(type) ||
        dynamic_cast<const VectorType *>(type))
        basePtr = arrayOrVector->GetLValue(ctx);
    else {
        type = type->GetReferenceTarget();
        assert(dynamic_cast<const ArrayType *>(type) ||
               dynamic_cast<const VectorType *>(type));
        basePtr = arrayOrVector->GetValue(ctx);
    }
    if (!basePtr)
        return NULL;

    basePtr = lCastUniformVectorBasePtr(basePtr, ctx);

    ctx->SetDebugPos(pos);
    return ctx->GetElementPtrInst(basePtr, LLVMInt32(0), index->GetValue(ctx));
}


Expr *
IndexExpr::Optimize() {
    if (arrayOrVector) 
        arrayOrVector = arrayOrVector->Optimize();
    if (index) 
        index = index->Optimize();
    if (arrayOrVector == NULL || index == NULL)
        return NULL;

    return this;
}


Expr *
IndexExpr::TypeCheck() {
    if (arrayOrVector) 
        arrayOrVector = arrayOrVector->TypeCheck();
    if (index) 
        index = index->TypeCheck();
    
    if (!arrayOrVector || !index || !index->GetType())
        return NULL;

    const Type *arrayOrVectorType = arrayOrVector->GetType();
    if (!arrayOrVectorType)
        return NULL;

    if (dynamic_cast<const SequentialType *>(arrayOrVectorType->GetReferenceTarget()) == NULL) {
        Error(pos, "Trying to index into non-array or vector type \"%s\".", 
              arrayOrVectorType->GetString().c_str());
        return NULL;
    }

    bool isUniform = (index->GetType()->IsUniformType() && 
                      !g->opt.disableUniformMemoryOptimizations);
    const Type *indexType = isUniform ? AtomicType::UniformInt32 : 
                                        AtomicType::VaryingInt32;
    index = index->TypeConv(indexType, "array index");
    if (!index)
        return NULL;

    return this;
}


void
IndexExpr::Print() const {
    if (!arrayOrVector || !index || !GetType())
        return;

    printf("[%s] index ", GetType()->GetString().c_str());
    arrayOrVector->Print();
    printf("[");
    index->Print();
    printf("]");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// MemberExpr

/** Map one character ids to vector element numbers.  Allow a few different
    conventions--xyzw, rgba, uv.
 */
static int
lIdentifierToVectorElement(char id) {
    switch (id) {
    case 'x':
    case 'r':
    case 'u':
        return 0;
    case 'y':
    case 'g':
    case 'v':
        return 1;
    case 'z':
    case 'b':
        return 2;
    case 'w':
    case 'a':
        return 3;
    default:
        return -1;
    }
}

class StructMemberExpr : public MemberExpr
{
public:
    StructMemberExpr(Expr *e, const char *id, SourcePos p,
                     SourcePos idpos, const StructType* structType);

    const Type* GetType() const;

    int getElementNumber() const;

private:
    const StructType* exprStructType;
};

StructMemberExpr::StructMemberExpr(Expr *e, const char *id, SourcePos p,
                                   SourcePos idpos,
                                   const StructType* structType)
    : MemberExpr(e, id, p, idpos), exprStructType(structType) {
}

const Type*
StructMemberExpr::GetType() const {
    // It's a struct, and the result type is the element
    // type, possibly promoted to varying if the struct type / lvalue
    // is varying.
    const Type *elementType = exprStructType->GetElementType(identifier);
    if (!elementType)
        Error(identifierPos,
              "Element name \"%s\" not present in struct type \"%s\".%s",
              identifier.c_str(), exprStructType->GetString().c_str(),
              getCandidateNearMatches().c_str());

    if (exprStructType->IsVaryingType()) 
        return elementType->GetAsVaryingType();
    else
        return elementType;
}

int
StructMemberExpr::getElementNumber() const {
    int elementNumber = exprStructType->GetElementNumber(identifier);
    if (elementNumber == -1)
        Error(identifierPos,
              "Element name \"%s\" not present in struct type \"%s\".%s",
              identifier.c_str(), exprStructType->GetString().c_str(),
              getCandidateNearMatches().c_str());
    return elementNumber;
}

class VectorMemberExpr : public MemberExpr
{
public:
    VectorMemberExpr(Expr *e, const char *id, SourcePos p,
                     SourcePos idpos, const VectorType* vectorType);

    ~VectorMemberExpr();

    const Type* GetType() const;

    llvm::Value* GetLValue(FunctionEmitContext* ctx) const;

    llvm::Value* GetValue(FunctionEmitContext* ctx) const;

    int getElementNumber() const;
private:
    const VectorType* exprVectorType;
    const VectorType* memberType;
};

VectorMemberExpr::VectorMemberExpr(Expr *e, const char *id, SourcePos p,
                                   SourcePos idpos,
                                   const VectorType* vectorType)
    : MemberExpr(e, id, p, idpos), exprVectorType(vectorType) {
    memberType = new VectorType(exprVectorType->GetElementType(),
                                identifier.length());
}

VectorMemberExpr::~VectorMemberExpr() {
    delete memberType;
}

const Type*
VectorMemberExpr::GetType() const {
    // For 1-element expressions, we have the base vector element
    // type.  For n-element expressions, we have a shortvec type
    // with n > 1 elements.  This can be changed when we get
    // type<1> -> type conversions.
    if (identifier.length() == 1) {
        return exprVectorType->GetElementType();
    } else {
        return memberType;
    }
}

llvm::Value*
VectorMemberExpr::GetLValue(FunctionEmitContext* ctx) const {
    if (identifier.length() == 1) {
        return MemberExpr::GetLValue(ctx);
    } else {
        return NULL;
    }
}

llvm::Value*
VectorMemberExpr::GetValue(FunctionEmitContext* ctx) const {
    if (identifier.length() == 1) {
        return MemberExpr::GetValue(ctx);
    } else {
        std::vector<int> indices;

        for (size_t i = 0; i < identifier.size(); ++i) {
            int idx = lIdentifierToVectorElement(identifier[i]);
            if (idx == -1)
                Error(pos,
                      "Invalid swizzle charcter '%c' in swizzle \"%s\".",
                      identifier[i], identifier.c_str());

            indices.push_back(idx);
        }

        llvm::Value *basePtr = expr->GetLValue(ctx);
        if (basePtr == NULL) {
            assert(m->errorCount > 0);
            return NULL;
        }
        llvm::Value *ltmp = ctx->AllocaInst(memberType->LLVMType(g->ctx), 
                                            "vector_tmp");

        ctx->SetDebugPos(pos);
        for (size_t i = 0; i < identifier.size(); ++i) {
            llvm::Value *ptmp =
                ctx->GetElementPtrInst(ltmp, 0, i, "new_offset");
            llvm::Value *initLValue =
                ctx->GetElementPtrInst(basePtr , 0,
                                       indices[i], "orig_offset");
            llvm::Value *initValue =
                ctx->LoadInst(initLValue, memberType->GetElementType(),
                              "vec_element");
            ctx->StoreInst(initValue, ptmp);
        }

        return ctx->LoadInst(ltmp, memberType, "swizzle_vec");
    }
}

int
VectorMemberExpr::getElementNumber() const {
    int elementNumber = lIdentifierToVectorElement(identifier[0]);
    if (elementNumber == -1)
        Error(pos, "Vector element identifier \"%s\" unknown.", 
              identifier.c_str());
    return elementNumber;
}

class ReferenceMemberExpr : public MemberExpr
{
public:
    ReferenceMemberExpr(Expr *e, const char *id, SourcePos p,
                        SourcePos idpos, const ReferenceType* referenceType);

    const Type* GetType() const;

    int getElementNumber() const;

    llvm::Value* GetLValue(FunctionEmitContext* ctx) const;

private:
    const ReferenceType* exprReferenceType;
    MemberExpr* dereferencedExpr;
};

ReferenceMemberExpr::ReferenceMemberExpr(Expr *e, const char *id, SourcePos p,
                                         SourcePos idpos,
                                         const ReferenceType* referenceType)
    : MemberExpr(e, id, p, idpos), exprReferenceType(referenceType) {
    const Type* refTarget = exprReferenceType->GetReferenceTarget();
    const StructType* structType
        = dynamic_cast<const StructType *>(refTarget);
    const VectorType* vectorType
        = dynamic_cast<const VectorType *>(refTarget);

    if (structType != NULL) {
        dereferencedExpr = new StructMemberExpr(e, id, p, idpos, structType);
    } else if (vectorType != NULL) {
        dereferencedExpr = new VectorMemberExpr(e, id, p, idpos, vectorType);
    } else {
        dereferencedExpr = NULL;
    }
}

const Type*
ReferenceMemberExpr::GetType() const {
    if (dereferencedExpr == NULL) {
        Error(pos, "Can't access member of non-struct/vector type \"%s\".",
              exprReferenceType->GetString().c_str());
        return NULL;
    } else {
        return dereferencedExpr->GetType();
    }
}

int
ReferenceMemberExpr::getElementNumber() const {
    if (dereferencedExpr == NULL) {
        // FIXME: I think we shouldn't ever get here and that
        // typechecking should have caught this case
        return -1;
    } else {
        return dereferencedExpr->getElementNumber();
    }
}

llvm::Value*
ReferenceMemberExpr::GetLValue(FunctionEmitContext* ctx) const {
    if (dereferencedExpr == NULL) {
        // FIXME: again I think typechecking should have caught this
        Error(pos, "Can't access member of non-struct/vector type \"%s\".",
              exprReferenceType->GetString().c_str());
        return NULL;
    }

    //FIXME: Minor Code-dup...this is the same as the base, except
    // llvm::Value *basePtr = expr->GetLValue instead of expr->getValue
    llvm::Value *basePtr = expr->GetValue(ctx);
    if (!basePtr)
        return NULL;

    int elementNumber = getElementNumber();
    if (elementNumber == -1)
        return NULL;

    ctx->SetDebugPos(pos);
    return ctx->GetElementPtrInst(basePtr, 0, elementNumber);
}


MemberExpr*
MemberExpr::create(Expr *e, const char *id, SourcePos p, SourcePos idpos) {
    const Type* exprType;
    if (e == NULL || (exprType = e->GetType()) == NULL)
        return new MemberExpr(e, id, p, idpos);

    const StructType* structType = dynamic_cast<const StructType*>(exprType);
    if (structType != NULL)
        return new StructMemberExpr(e, id, p, idpos, structType);

    const VectorType* vectorType = dynamic_cast<const VectorType*>(exprType);
    if (vectorType != NULL)
        return new VectorMemberExpr(e, id, p, idpos, vectorType);

    const ReferenceType* referenceType = dynamic_cast<const ReferenceType*>(exprType);
    if (referenceType != NULL)
        return new ReferenceMemberExpr(e, id, p, idpos, referenceType);
  
    return new MemberExpr(e, id, p, idpos);
}

MemberExpr::MemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos) 
    : Expr(p), identifierPos(idpos) {
    expr = e;
    identifier = id;
}


llvm::Value *
MemberExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr) 
        return NULL;

    llvm::Value *lvalue = GetLValue(ctx);
    if (!lvalue) {
        // As in the array case, this may be a temporary that hasn't hit
        // memory; get the full value and stuff it into a temporary array
        // so that we can index from there...
        llvm::Value *val = expr->GetValue(ctx);
        if (!val) {
            assert(m->errorCount > 0);
            return NULL;
        }
        ctx->SetDebugPos(pos);
        const Type *exprType = expr->GetType();
        llvm::Value *ptr = ctx->AllocaInst(exprType->LLVMType(g->ctx), 
                                           "struct_tmp");
        ctx->StoreInst(val, ptr);

        int elementNumber = getElementNumber();
        if (elementNumber == -1)
            return NULL;
        lvalue = ctx->GetElementPtrInst(ptr, 0, elementNumber);
    }

    ctx->SetDebugPos(pos);
    return ctx->LoadInst(lvalue, GetType(), "structelement");
}


const Type *
MemberExpr::GetType() const {
    return NULL;
}


Symbol *
MemberExpr::GetBaseSymbol() const {
    return expr ? expr->GetBaseSymbol() : NULL;
}


int
MemberExpr::getElementNumber() const {
    return -1;
}


llvm::Value *
MemberExpr::GetLValue(FunctionEmitContext *ctx) const {
    //This kindof feels like magic, but this functionality
    // will have to be overridden in VectorMemberExpr when
    // we support multi-swizzle.
    const Type *exprType;
    if (!expr || ((exprType = expr->GetType()) == NULL))
        return NULL;

    ctx->SetDebugPos(pos);
    llvm::Value *basePtr = expr->GetLValue(ctx);
    if (!basePtr)
        return NULL;

    int elementNumber = getElementNumber();
    if (elementNumber == -1)
        return NULL;

    ctx->SetDebugPos(pos);
    return ctx->GetElementPtrInst(basePtr, 0, elementNumber);
}


Expr *
MemberExpr::TypeCheck() {
    if (expr) 
        expr = expr->TypeCheck();
    return expr ? this : NULL;
}


Expr *
MemberExpr::Optimize() {
    if (expr) 
        expr = expr->Optimize();
    return expr ? this : NULL;
}


void
MemberExpr::Print() const {
    if (!expr || !GetType())
        return;

    printf("[%s] member (", GetType()->GetString().c_str());
    expr->Print();
    printf(" . %s)", identifier.c_str());
    pos.Print();
}


/** There is no structure member with the name we've got in "identifier".
    Use the approximate string matching routine to see if the identifier is
    a minor misspelling of one of the ones that is there.
 */
std::string
MemberExpr::getCandidateNearMatches() const {
    const StructType *structType = 
        dynamic_cast<const StructType *>(expr->GetType());
    if (!structType)
        return "";

    std::vector<std::string> elementNames;
    for (int i = 0; i < structType->GetElementCount(); ++i)
        elementNames.push_back(structType->GetElementName(i));
    std::vector<std::string> alternates = MatchStrings(identifier, elementNames);
    if (!alternates.size())
        return "";

    std::string ret = " Did you mean ";
    for (unsigned int i = 0; i < alternates.size(); ++i) {
        ret += std::string("\"") + alternates[i] + std::string("\"");
        if (i < alternates.size() - 1) ret += ", or ";
    }
    ret += "?";
    return ret;
}


///////////////////////////////////////////////////////////////////////////
// ConstExpr

ConstExpr::ConstExpr(const Type *t, int8_t i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt8);
    int8Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int8_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt8 || 
           type == AtomicType::VaryingConstInt8);
    for (int j = 0; j < Count(); ++j)
        int8Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint8_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformUInt8);
    uint8Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint8_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstUInt8 || 
           type == AtomicType::VaryingConstUInt8);
    for (int j = 0; j < Count(); ++j)
        uint8Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, int16_t i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt16);
    int16Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int16_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt16 || 
           type == AtomicType::VaryingConstInt16);
    for (int j = 0; j < Count(); ++j)
        int16Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint16_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformUInt16);
    uint16Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint16_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstUInt16 || 
           type == AtomicType::VaryingConstUInt16);
    for (int j = 0; j < Count(); ++j)
        uint16Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, int32_t i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt32);
    int32Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int32_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt32 || 
           type == AtomicType::VaryingConstInt32);
    for (int j = 0; j < Count(); ++j)
        int32Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint32_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstUInt32 ||
           (dynamic_cast<const EnumType *>(type) != NULL &&
            type->IsUniformType()));
    uint32Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint32_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstUInt32 || 
           type == AtomicType::VaryingConstUInt32 ||
           (dynamic_cast<const EnumType *>(type) != NULL));
    for (int j = 0; j < Count(); ++j)
        uint32Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, float f, SourcePos p)
    : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstFloat);
    floatVal[0] = f;
}


ConstExpr::ConstExpr(const Type *t, float *f, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstFloat || 
           type == AtomicType::VaryingConstFloat);
    for (int j = 0; j < Count(); ++j)
        floatVal[j] = f[j];
}


ConstExpr::ConstExpr(const Type *t, int64_t i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt64);
    int64Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int64_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstInt64 || 
           type == AtomicType::VaryingConstInt64);
    for (int j = 0; j < Count(); ++j)
        int64Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint64_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformUInt64);
    uint64Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint64_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstUInt64 || 
           type == AtomicType::VaryingConstUInt64);
    for (int j = 0; j < Count(); ++j)
        uint64Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, double f, SourcePos p)
    : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstDouble);
    doubleVal[0] = f;
}


ConstExpr::ConstExpr(const Type *t, double *f, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstDouble || 
           type == AtomicType::VaryingConstDouble);
    for (int j = 0; j < Count(); ++j)
        doubleVal[j] = f[j];
}


ConstExpr::ConstExpr(const Type *t, bool b, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstBool);
    boolVal[0] = b;
}


ConstExpr::ConstExpr(const Type *t, bool *b, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    assert(type == AtomicType::UniformConstBool || 
           type == AtomicType::VaryingConstBool);
    for (int j = 0; j < Count(); ++j)
        boolVal[j] = b[j];
}


ConstExpr::ConstExpr(ConstExpr *old, double *v) 
    : Expr(old->pos) {
    type = old->type;

    AtomicType::BasicType basicType = getBasicType();

    switch (basicType) {
    case AtomicType::TYPE_BOOL:
        for (int i = 0; i < Count(); ++i)
            boolVal[i] = (v[i] != 0.);
        break;
    case AtomicType::TYPE_INT8:
        for (int i = 0; i < Count(); ++i)
            int8Val[i] = (int)v[i];
        break;
    case AtomicType::TYPE_UINT8:
        for (int i = 0; i < Count(); ++i)
            uint8Val[i] = (unsigned int)v[i];
        break;
    case AtomicType::TYPE_INT16:
        for (int i = 0; i < Count(); ++i)
            int16Val[i] = (int)v[i];
        break;
    case AtomicType::TYPE_UINT16:
        for (int i = 0; i < Count(); ++i)
            uint16Val[i] = (unsigned int)v[i];
        break;
    case AtomicType::TYPE_INT32:
        for (int i = 0; i < Count(); ++i)
            int32Val[i] = (int)v[i];
        break;
    case AtomicType::TYPE_UINT32:
        for (int i = 0; i < Count(); ++i)
            uint32Val[i] = (unsigned int)v[i];
        break;
    case AtomicType::TYPE_FLOAT:
        for (int i = 0; i < Count(); ++i)
            floatVal[i] = (float)v[i];
        break;
    case AtomicType::TYPE_DOUBLE:
        for (int i = 0; i < Count(); ++i)
            doubleVal[i] = v[i];
        break;
    case AtomicType::TYPE_INT64:
    case AtomicType::TYPE_UINT64:
        // For now, this should never be reached 
        FATAL("fixme; we need another constructor so that we're not trying to pass "
               "double values to init an int64 type...");
    default:
        FATAL("unimplemented const type");
    }
}


AtomicType::BasicType
ConstExpr::getBasicType() const {
    const AtomicType *at = dynamic_cast<const AtomicType *>(type);
    if (at != NULL)
        return at->basicType;
    else {
        assert(dynamic_cast<const EnumType *>(type) != NULL);
        return AtomicType::TYPE_UINT32;
    }
}


const Type *
ConstExpr::GetType() const { 
    return type; 
}


llvm::Value *
ConstExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    bool isVarying = type->IsVaryingType();

    AtomicType::BasicType basicType = getBasicType();

    switch (basicType) {
    case AtomicType::TYPE_BOOL:
        if (isVarying)
            return LLVMBoolVector(boolVal);
        else
            return boolVal[0] ? LLVMTrue : LLVMFalse;
    case AtomicType::TYPE_INT8:
        return isVarying ? LLVMInt8Vector(int8Val) : 
                           LLVMInt8(int8Val[0]);
    case AtomicType::TYPE_UINT8:
        return isVarying ? LLVMUInt8Vector(uint8Val) : 
                           LLVMUInt8(uint8Val[0]);
    case AtomicType::TYPE_INT16:
        return isVarying ? LLVMInt16Vector(int16Val) : 
                           LLVMInt16(int16Val[0]);
    case AtomicType::TYPE_UINT16:
        return isVarying ? LLVMUInt16Vector(uint16Val) : 
                           LLVMUInt16(uint16Val[0]);
    case AtomicType::TYPE_INT32:
        return isVarying ? LLVMInt32Vector(int32Val) : 
                           LLVMInt32(int32Val[0]);
    case AtomicType::TYPE_UINT32:
        return isVarying ? LLVMUInt32Vector(uint32Val) : 
                           LLVMUInt32(uint32Val[0]);
    case AtomicType::TYPE_FLOAT:
        return isVarying ? LLVMFloatVector(floatVal) : 
                           LLVMFloat(floatVal[0]);
    case AtomicType::TYPE_INT64:
        return isVarying ? LLVMInt64Vector(int64Val) : 
                           LLVMInt64(int64Val[0]);
    case AtomicType::TYPE_UINT64:
        return isVarying ? LLVMUInt64Vector(uint64Val) : 
                           LLVMUInt64(uint64Val[0]);
    case AtomicType::TYPE_DOUBLE:
        return isVarying ? LLVMDoubleVector(doubleVal) : 
                           LLVMDouble(doubleVal[0]);
    default:
        FATAL("unimplemented const type");
        return NULL;
    }
}


/* Type conversion templates: take advantage of C++ function overloading
   rules to get the one we want to match. */

/* First the most general case, just use C++ type conversion if nothing
   else matches */
template <typename From, typename To> static inline void
lConvertElement(From from, To *to) {
    *to = (To)from;
}


/** When converting from bool types to numeric types, make sure the result
    is one or zero.
    FIXME: this is a different rule than we use elsewhere, where we sign extend
    the bool.  We should fix the other case to just zero extend and then
    patch up places in the stdlib that depend on sign extension to call a 
    routine to make that happen.
 */ 
template <typename To> static inline void
lConvertElement(bool from, To *to) {
    *to = from ? (To)1 : (To)0;
}


/** When converting numeric types to bool, compare to zero.  (Do we
    actually need this one??) */
template <typename From> static inline void
lConvertElement(From from, bool *to) {
    *to = (from != 0);
}


/** And bool -> bool is just assignment */
static inline void
lConvertElement(bool from, bool *to) {
    *to = from;
}


/** Type conversion utility function
 */
template <typename From, typename To> static void
lConvert(const From *from, To *to, int count, bool forceVarying) {
    for (int i = 0; i < count; ++i)
        lConvertElement(from[i], &to[i]);

    if (forceVarying && count == 1)
        for (int i = 1; i < g->target.vectorWidth; ++i)
            to[i] = to[0];
}


int
ConstExpr::AsInt64(int64_t *ip, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, ip, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsUInt64(uint64_t *up, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, up, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsDouble(double *d, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   d, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   d, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  d, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  d, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, d, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  d, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, d, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  d, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, d, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  d, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, d, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsFloat(float *fp, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   fp, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   fp, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  fp, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  fp, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, fp, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  fp, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, fp, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  fp, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, fp, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  fp, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, fp, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsBool(bool *b, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   b, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   b, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  b, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  b, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, b, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  b, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, b, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  b, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, b, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  b, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, b, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsInt8(int8_t *ip, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, ip, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsUInt8(uint8_t *up, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, up, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsInt16(int16_t *ip, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, ip, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsUInt16(uint16_t *up, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, up, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsInt32(int32_t *ip, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, ip, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  ip, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, ip, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::AsUInt32(uint32_t *up, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:   lConvert(boolVal,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT8:   lConvert(int8Val,   up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT8:  lConvert(uint8Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT16:  lConvert(int16Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT16: lConvert(uint16Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT32:  lConvert(int32Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT32: lConvert(uint32Val, up, Count(), forceVarying); break;
    case AtomicType::TYPE_FLOAT:  lConvert(floatVal,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_DOUBLE: lConvert(doubleVal, up, Count(), forceVarying); break;
    case AtomicType::TYPE_INT64:  lConvert(int64Val,  up, Count(), forceVarying); break;
    case AtomicType::TYPE_UINT64: lConvert(uint64Val, up, Count(), forceVarying); break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}


int
ConstExpr::Count() const { 
    return GetType()->IsVaryingType() ? g->target.vectorWidth : 1; 
}


llvm::Constant *
ConstExpr::GetConstant(const Type *type) const {
    // Caller shouldn't be trying to stuff a varying value here into a
    // constant type.
    if (type->IsUniformType())
        assert(Count() == 1);

    type = type->GetAsNonConstType();
    if (type == AtomicType::UniformBool || type == AtomicType::VaryingBool) {
        bool bv[ISPC_MAX_NVEC];
        AsBool(bv, type->IsVaryingType());
        if (type->IsUniformType())
            return bv[0] ? LLVMTrue : LLVMFalse;
        else
            return LLVMBoolVector(bv);
    }
    else if (type == AtomicType::UniformInt8 || type == AtomicType::VaryingInt8) {
        int8_t iv[ISPC_MAX_NVEC];
        AsInt8(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt8(iv[0]);
        else
            return LLVMInt8Vector(iv);
    }
    else if (type == AtomicType::UniformUInt8 || type == AtomicType::VaryingUInt8 ||
             dynamic_cast<const EnumType *>(type) != NULL) {
        uint8_t uiv[ISPC_MAX_NVEC];
        AsUInt8(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt8(uiv[0]);
        else
            return LLVMUInt8Vector(uiv);
    }
    else if (type == AtomicType::UniformInt16 || type == AtomicType::VaryingInt16) {
        int16_t iv[ISPC_MAX_NVEC];
        AsInt16(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt16(iv[0]);
        else
            return LLVMInt16Vector(iv);
    }
    else if (type == AtomicType::UniformUInt16 || type == AtomicType::VaryingUInt16 ||
             dynamic_cast<const EnumType *>(type) != NULL) {
        uint16_t uiv[ISPC_MAX_NVEC];
        AsUInt16(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt16(uiv[0]);
        else
            return LLVMUInt16Vector(uiv);
    }
    else if (type == AtomicType::UniformInt32 || type == AtomicType::VaryingInt32) {
        int32_t iv[ISPC_MAX_NVEC];
        AsInt32(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt32(iv[0]);
        else
            return LLVMInt32Vector(iv);
    }
    else if (type == AtomicType::UniformUInt32 || type == AtomicType::VaryingUInt32 ||
             dynamic_cast<const EnumType *>(type) != NULL) {
        uint32_t uiv[ISPC_MAX_NVEC];
        AsUInt32(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt32(uiv[0]);
        else
            return LLVMUInt32Vector(uiv);
    }
    else if (type == AtomicType::UniformFloat || type == AtomicType::VaryingFloat) {
        float fv[ISPC_MAX_NVEC];
        AsFloat(fv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMFloat(fv[0]);
        else
            return LLVMFloatVector(fv);
    }
    else if (type == AtomicType::UniformInt64 || type == AtomicType::VaryingInt64) {
        int64_t iv[ISPC_MAX_NVEC];
        AsInt64(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt64(iv[0]);
        else
            return LLVMInt64Vector(iv);
    }
    else if (type == AtomicType::UniformUInt64 || type == AtomicType::VaryingUInt64) {
        uint64_t uiv[ISPC_MAX_NVEC];
        AsUInt64(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt64(uiv[0]);
        else
            return LLVMUInt64Vector(uiv);
    }
    else if (type == AtomicType::UniformDouble || type == AtomicType::VaryingDouble) {
        double dv[ISPC_MAX_NVEC];
        AsDouble(dv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMDouble(dv[0]);
        else
            return LLVMDoubleVector(dv);
    }
    else {
        FATAL("unexpected type in ConstExpr::GetConstant()");
        return NULL;
    }
}


Expr *
ConstExpr::Optimize() {
    return this;
}


Expr *
ConstExpr::TypeCheck() {
    return this;
}


void
ConstExpr::Print() const {
    printf("[%s] (", GetType()->GetString().c_str());
    for (int i = 0; i < Count(); ++i) {
        switch (getBasicType()) {
        case AtomicType::TYPE_BOOL:
            printf("%s", boolVal[i] ? "true" : "false");
            break;
        case AtomicType::TYPE_INT8:
            printf("%d", (int)int8Val[i]);
            break;
        case AtomicType::TYPE_UINT8:
            printf("%u", (int)uint8Val[i]);
            break;
        case AtomicType::TYPE_INT16:
            printf("%d", (int)int16Val[i]);
            break;
        case AtomicType::TYPE_UINT16:
            printf("%u", (int)uint16Val[i]);
            break;
        case AtomicType::TYPE_INT32:
            printf("%d", int32Val[i]);
            break;
        case AtomicType::TYPE_UINT32:
            printf("%u", uint32Val[i]);
            break;
        case AtomicType::TYPE_FLOAT:
            printf("%f", floatVal[i]);
            break;
        case AtomicType::TYPE_INT64:
#ifdef ISPC_IS_LINUX
            printf("%ld", int64Val[i]);
#else
            printf("%lld", int64Val[i]);
#endif
            break;
        case AtomicType::TYPE_UINT64:
#ifdef ISPC_IS_LINUX
            printf("%lu", uint64Val[i]);
#else
            printf("%llu", uint64Val[i]);
#endif
            break;
        case AtomicType::TYPE_DOUBLE:
            printf("%f", doubleVal[i]);
            break;
        default:
            FATAL("unimplemented const type");
        }
        if (i != Count() - 1)
            printf(", ");
    }
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// TypeCastExpr

TypeCastExpr::TypeCastExpr(const Type *t, Expr *e, SourcePos p) 
  : Expr(p) {
    type = t;
    expr = e;
}


/** Handle all the grungy details of type conversion between atomic types.
    Given an input value in exprVal of type fromType, convert it to the
    llvm::Value with type toType.
 */
static llvm::Value *
lTypeConvAtomic(FunctionEmitContext *ctx, llvm::Value *exprVal, 
                const AtomicType *toType, const AtomicType *fromType,
                SourcePos pos) {
    llvm::Value *cast = NULL;

    switch (toType->basicType) {
    case AtomicType::TYPE_FLOAT: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::FloatType : 
                                        LLVMTypes::FloatVectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                // If we have a bool vector of i32 element,s first truncate
                // down to a single bit
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            // And then do an unisgned int->float cast
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, "bool2float");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_INT64:
            cast = ctx->CastInst(llvm::Instruction::SIToFP, // signed int to float
                                 exprVal, targetType, "int2float");
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_UINT64:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from unsigned int to float is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float
                                 exprVal, targetType, "uint2float");
            break;
        case AtomicType::TYPE_FLOAT:
            // No-op cast.
            cast = exprVal;
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->FPCastInst(exprVal, targetType, "double2float");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_DOUBLE: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::DoubleType :
                                        LLVMTypes::DoubleVectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                // truncate i32 bool vector values to i1s
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to double
                                 exprVal, targetType, "bool2double");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_INT64:
            cast = ctx->CastInst(llvm::Instruction::SIToFP, // signed int
                                 exprVal, targetType, "int2double");
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_UINT64:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from unsigned int64 to float is slow. "
                                   "Use \"int64\" if possible");
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, "uint2double");
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->FPCastInst(exprVal, targetType, "float2double");
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = exprVal;
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT8: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int8Type :
                                        LLVMTypes::Int8VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2int");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_UINT8:
            cast = exprVal;
            break;
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, "int64_to_int8");
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "float2int");
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "double2int");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT8: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int8Type :
                                        LLVMTypes::Int8VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2uint");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_UINT8:
            cast = exprVal;
            break;
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, "int64_to_uint8");
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from float to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, "float2uint");
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from double to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, "double2uint");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT16: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int16Type :
                                        LLVMTypes::Int16VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2int");
            break;
        case AtomicType::TYPE_INT8:
            cast = ctx->SExtInst(exprVal, targetType, "int2int16");
            break;
        case AtomicType::TYPE_UINT8:
            cast = ctx->ZExtInst(exprVal, targetType, "uint2uint16");
            break;
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "float2int");
            break;
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, "int64_to_int16");
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "double2int");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT16: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int16Type :
                                        LLVMTypes::Int16VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2uint16");
            break;
        case AtomicType::TYPE_INT8:
            cast = ctx->SExtInst(exprVal, targetType, "uint2uint16");
            break;
        case AtomicType::TYPE_UINT8:
            cast = ctx->ZExtInst(exprVal, targetType, "uint2uint16");
            break;            
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from float to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, "float2uint");
            break;
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, "int64_to_uint16");
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from double to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, "double2uint");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT32: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int32Type :
                                        LLVMTypes::Int32VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2int");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
            cast = ctx->SExtInst(exprVal, targetType, "int2int32");
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
            cast = ctx->ZExtInst(exprVal, targetType, "uint2uint32");
            break;
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "float2int");
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, "int64_to_int32");
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "double2int");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT32: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int32Type :
                                        LLVMTypes::Int32VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2uint");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
            cast = ctx->SExtInst(exprVal, targetType, "uint2uint");
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
            cast = ctx->ZExtInst(exprVal, targetType, "uint2uint");
            break;            
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from float to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, "float2uint");
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, "int64_to_uint32");
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from double to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, "double2uint");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT64: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int64Type : 
                                        LLVMTypes::Int64VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() &&
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2int64");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
            cast = ctx->SExtInst(exprVal, targetType, "int_to_int64");
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
            cast = ctx->ZExtInst(exprVal, targetType, "uint_to_int64");
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "float2int64");
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = exprVal;
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, "double2int64");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT64: {
        LLVM_TYPE_CONST llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int64Type : 
                                        LLVMTypes::Int64VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, "bool_to_i1");
            cast = ctx->ZExtInst(exprVal, targetType, "bool2uint");
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
            cast = ctx->SExtInst(exprVal, targetType, "int_to_uint64");
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
            cast = ctx->ZExtInst(exprVal, targetType, "uint_to_uint64");
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from float to unsigned int64 is slow. "
                                   "Use \"int64\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // signed int
                                 exprVal, targetType, "float2uint");
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = exprVal;
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType())
                PerformanceWarning(pos, "Conversion from double to unsigned int64 is slow. "
                                   "Use \"int64\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // signed int
                                 exprVal, targetType, "double2uint");
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_BOOL: {
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            cast = exprVal;
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_UINT8: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMInt8(0) : 
                (llvm::Value *)LLVMInt8Vector((int8_t)0);
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                exprVal, zero, "cmpi0");
            break;
        }
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMInt16(0) : 
                (llvm::Value *)LLVMInt16Vector((int16_t)0);
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                exprVal, zero, "cmpi0");
            break;
        }
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMInt32(0) : 
                (llvm::Value *)LLVMInt32Vector(0);
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                exprVal, zero, "cmpi0");
            break;
        }
        case AtomicType::TYPE_FLOAT: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMFloat(0.f) : 
                (llvm::Value *)LLVMFloatVector(0.f);
            cast = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_ONE,
                                exprVal, zero, "cmpf0");
            break;
        }
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMInt64(0) : 
                (llvm::Value *)LLVMInt64Vector((int64_t)0);
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                exprVal, zero, "cmpi0");
            break;
        }
        case AtomicType::TYPE_DOUBLE: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMDouble(0.) : 
                (llvm::Value *)LLVMDoubleVector(0.);
            cast = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_ONE,
                                exprVal, zero, "cmpd0");
            break;
        }
        default:
            FATAL("unimplemented");
        }

        if (fromType->IsUniformType()) {
            if (toType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType) {
                // extend out to i32 bool values from i1 here.  then we'll
                // turn into a vector below, the way it does for everyone
                // else...
                cast = ctx->SExtInst(cast, LLVMTypes::BoolVectorType->getElementType(),
                                     "i1bool_to_i32bool");
            }
        }
        else
            // fromType->IsVaryingType())
            cast = ctx->I1VecToBoolVec(cast);

        break;
    }
    default:
        FATAL("unimplemented");
    }

    // If we also want to go from uniform to varying, replicate out the
    // value across the vector elements..
    if (toType->IsVaryingType() && fromType->IsUniformType()) {
        LLVM_TYPE_CONST llvm::Type *vtype = toType->LLVMType(g->ctx);
        llvm::Value *castVec = llvm::UndefValue::get(vtype);
        for (int i = 0; i < g->target.vectorWidth; ++i)
            castVec = ctx->InsertInst(castVec, cast, i, "smearinsert");
        return castVec;
    }
    else
        return cast;
}


/** Converts the given value of the given type to be the varying
    equivalent, returning the resulting value.
 */
static llvm::Value *
lUniformValueToVarying(FunctionEmitContext *ctx, llvm::Value *value,
                       const Type *type) {
    // nothing to do if it's already varying
    if (type->IsVaryingType())
        return value;

    LLVM_TYPE_CONST llvm::Type *llvmType = type->GetAsVaryingType()->LLVMType(g->ctx);
    llvm::Value *retValue = llvm::UndefValue::get(llvmType);

    // for structs/arrays/vectors, just recursively make their elements
    // varying (if needed) and populate the return value.
    const CollectionType *collectionType = 
        dynamic_cast<const CollectionType *>(type);
    if (collectionType != NULL) {
        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            llvm::Value *v = ctx->ExtractInst(value, i, "get_element");
            v = lUniformValueToVarying(ctx, v, collectionType->GetElementType(i));
            retValue = ctx->InsertInst(retValue, v, i, "set_element");
        }
        return retValue;
    }

    // Otherwise we must have a uniform AtomicType, so smear its value
    // across the vector lanes.
    assert(dynamic_cast<const AtomicType *>(type) != NULL);
    for (int i = 0; i < g->target.vectorWidth; ++i)
        retValue = ctx->InsertInst(retValue, value, i, "smearinsert");
    return retValue;
}



llvm::Value *
TypeCastExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr)
        return NULL;

    ctx->SetDebugPos(pos);
    const Type *toType = GetType(), *fromType = expr->GetType();
    if (!toType || !fromType || toType == AtomicType::Void || 
        fromType == AtomicType::Void)
        // an error should have been issued elsewhere in this case
        return NULL;

    if (Type::Equal(toType->GetAsConstType(), fromType->GetAsConstType()))
        // There's nothing to do, just return the value.  (LLVM's type
        // system doesn't worry about constiness.)
        return expr->GetValue(ctx);

    // This also should be caught during typechecking
    assert(!(toType->IsUniformType() && fromType->IsVaryingType()));

    const ReferenceType *toReference = dynamic_cast<const ReferenceType *>(toType);
    const ReferenceType *fromReference = dynamic_cast<const ReferenceType *>(fromType);
    if (toReference && fromReference) {
        const Type *toTarget = toReference->GetReferenceTarget();
        const Type *fromTarget = fromReference->GetReferenceTarget();

        const ArrayType *toArray = dynamic_cast<const ArrayType *>(toTarget);
        const ArrayType *fromArray = dynamic_cast<const ArrayType *>(fromTarget);
        if (toArray && fromArray) {
            // cast array pointer from [n x foo] to [0 x foo] if needed to be able
            // to pass to a function that takes an unsized array as a parameter
            if(toArray->GetElementCount() != 0 && 
               (toArray->GetElementCount() != fromArray->GetElementCount()))
                Warning(pos, "Type-converting array of length %d to length %d",
                        fromArray->GetElementCount(), toArray->GetElementCount());
            assert(Type::Equal(toArray->GetBaseType()->GetAsConstType(),
                               fromArray->GetBaseType()->GetAsConstType()));
            llvm::Value *v = expr->GetValue(ctx);
            LLVM_TYPE_CONST llvm::Type *ptype = toType->LLVMType(g->ctx);
            return ctx->BitCastInst(v, ptype); //, "array_cast_0size");
        }

        assert(Type::Equal(toTarget, fromTarget) ||
               Type::Equal(toTarget, fromTarget->GetAsConstType()));
        return expr->GetValue(ctx);
    }

    const StructType *toStruct = dynamic_cast<const StructType *>(toType);
    const StructType *fromStruct = dynamic_cast<const StructType *>(fromType);
    if (toStruct && fromStruct) {
        // The only legal type conversions for structs are to go from a
        // uniform to a varying instance of the same struct type.
        assert(toStruct->IsVaryingType() && fromStruct->IsUniformType() &&
               Type::Equal(toStruct, fromStruct->GetAsVaryingType()));

        llvm::Value *origValue = expr->GetValue(ctx);
        if (!origValue)
            return NULL;
        return lUniformValueToVarying(ctx, origValue, fromType);
    }

    const VectorType *toVector = dynamic_cast<const VectorType *>(toType);
    const VectorType *fromVector = dynamic_cast<const VectorType *>(fromType);
    if (toVector && fromVector) {
        // this should be caught during typechecking
        assert(toVector->GetElementCount() == fromVector->GetElementCount());

        llvm::Value *exprVal = expr->GetValue(ctx);
        if (!exprVal)
            return NULL;

        // Emit instructions to do type conversion of each of the elements
        // of the vector.
        // FIXME: since uniform vectors are represented as
        // llvm::VectorTypes, we should just be able to issue the
        // corresponding vector type convert, which should be more
        // efficient by avoiding serialization!
        llvm::Value *cast = llvm::UndefValue::get(toType->LLVMType(g->ctx));
        for (int i = 0; i < toVector->GetElementCount(); ++i) {
            llvm::Value *ei = ctx->ExtractInst(exprVal, i);

            llvm::Value *conv = lTypeConvAtomic(ctx, ei, toVector->GetElementType(),
                                                fromVector->GetElementType(), pos);
            if (!conv) 
                return NULL;
            cast = ctx->InsertInst(cast, conv, i);
        }
        return cast;
    }

    llvm::Value *exprVal = expr->GetValue(ctx);
    if (!exprVal)
        return NULL;

    const EnumType *fromEnum = dynamic_cast<const EnumType *>(fromType);
    const EnumType *toEnum = dynamic_cast<const EnumType *>(toType);
    if (fromEnum)
        // treat it as an uint32 type for the below and all will be good.
        fromType = fromEnum->IsUniformType() ? AtomicType::UniformUInt32 :
            AtomicType::VaryingUInt32;
    if (toEnum)
        // treat it as an uint32 type for the below and all will be good.
        toType = toEnum->IsUniformType() ? AtomicType::UniformUInt32 :
            AtomicType::VaryingUInt32;

    const AtomicType *fromAtomic = dynamic_cast<const AtomicType *>(fromType);
    // at this point, coming from an atomic type is all that's left...
    assert(fromAtomic != NULL);

    if (toVector) {
        // scalar -> short vector conversion
        llvm::Value *conv = lTypeConvAtomic(ctx, exprVal, toVector->GetElementType(),
                                            fromAtomic, pos);
        if (!conv) 
            return NULL;

        llvm::Value *cast = llvm::UndefValue::get(toType->LLVMType(g->ctx));
        for (int i = 0; i < toVector->GetElementCount(); ++i)
            cast = ctx->InsertInst(cast, conv, i);
        return cast;
    }
    else {
        const AtomicType *toAtomic = dynamic_cast<const AtomicType *>(toType);
        // typechecking should ensure this is the case
        assert(toAtomic != NULL);

        return lTypeConvAtomic(ctx, exprVal, toAtomic, fromAtomic, pos);
    }
}


const Type *
TypeCastExpr::GetType() const { 
    return type; 
}


Expr *
TypeCastExpr::TypeCheck() {
    if (expr != NULL) 
        expr = expr->TypeCheck();
    if (expr == NULL)
        return NULL;

    const Type *toType = GetType(), *fromType = expr->GetType();
    if (toType == NULL || fromType == NULL)
        return NULL;

    const char *toTypeString = toType->GetString().c_str();
    const char *fromTypeString = fromType->GetString().c_str();

    // It's an error to attempt to convert from varying to uniform
    if (toType->IsUniformType() && !fromType->IsUniformType()) {
        Error(pos, "Can't assign 'varying' value to 'uniform' type \"%s\".",
              toTypeString);
        return NULL;
    }

    // And any kind of void type in a type cast doesn't make sense
    if (toType == AtomicType::Void || fromType == AtomicType::Void) {
        Error(pos, "Void type illegal in type cast from type \"%s\" to "
              "type \"%s\".", fromTypeString, toTypeString);
        return NULL;
    }

    // FIXME: do we need to worry more about references here?

    if (dynamic_cast<const VectorType *>(fromType) != NULL) {
        // Starting from a vector type; the result type must be a vector
        // type as well
        if (dynamic_cast<const VectorType *>(toType) == NULL) {
            Error(pos, "Can't convert vector type \"%s\" to non-vector type \"%s\".",
                  fromTypeString, toTypeString);
            return NULL;
        }

        // And the two vectors must have the same number of elements
        if (dynamic_cast<const VectorType *>(toType)->GetElementCount() != 
            dynamic_cast<const VectorType *>(fromType)->GetElementCount()) {
            Error(pos, "Can't convert vector type \"%s\" to differently-sized "
                  "vector type \"%s\".", fromTypeString, toTypeString);
            return NULL;
        }

        // And we're ok; since vectors can only hold AtomicTypes, we know
        // that type converting the elements will work.
        return this;
    }
    else if (dynamic_cast<const ArrayType *>(fromType) != NULL) {
        FATAL("Shouldn't ever get here");
        return this;
    }
    else {
        assert(dynamic_cast<const AtomicType *>(fromType) != NULL ||
               dynamic_cast<const EnumType *>(fromType) != NULL);
        // If we're going from an atomic or enum type, the only possible
        // result is another atomic or enum type
        if (dynamic_cast<const AtomicType *>(toType) == NULL &&
            dynamic_cast<const EnumType *>(toType) == NULL) {
            Error(pos, "Can't convert from type \"%s\" to \"%s\".",
                  fromTypeString, toTypeString);
            return NULL;
        }

        return this;
    }
}


Expr *
TypeCastExpr::Optimize() {
    if (expr != NULL) 
        expr = expr->Optimize();
    if (expr == NULL)
        return NULL;

    ConstExpr *constExpr = dynamic_cast<ConstExpr *>(expr);
    if (!constExpr)
        // We can't do anything if this isn't a const expr
        return this;

    const Type *toType = GetType();
    const AtomicType *toAtomic = dynamic_cast<const AtomicType *>(toType);
    const EnumType *toEnum = dynamic_cast<const EnumType *>(toType);
    // If we're not casting to an atomic or enum type, we can't do anything
    // here, since ConstExprs can only represent those two types.  (So
    // e.g. we're casting from an int to an int<4>.)
    if (toAtomic == NULL && toEnum == NULL)
        return this;

    bool forceVarying = toType->IsVaryingType();

    // All of the type conversion smarts we need is already in the
    // ConstExpr::AsBool(), etc., methods, so we just need to call the
    // appropriate one for the type that this cast is converting to.
    AtomicType::BasicType basicType = toAtomic ? toAtomic->basicType :
        AtomicType::TYPE_UINT32;
    switch (basicType) {
    case AtomicType::TYPE_BOOL: {
        bool bv[ISPC_MAX_NVEC];
        constExpr->AsBool(bv, forceVarying);
        return new ConstExpr(toType, bv, pos);
    }
    case AtomicType::TYPE_INT8: {
        int8_t iv[ISPC_MAX_NVEC];
        constExpr->AsInt8(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT8: {
        uint8_t uv[ISPC_MAX_NVEC];
        constExpr->AsUInt8(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_INT16: {
        int16_t iv[ISPC_MAX_NVEC];
        constExpr->AsInt16(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT16: {
        uint16_t uv[ISPC_MAX_NVEC];
        constExpr->AsUInt16(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_INT32: {
        int32_t iv[ISPC_MAX_NVEC];
        constExpr->AsInt32(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT32: {
        uint32_t uv[ISPC_MAX_NVEC];
        constExpr->AsUInt32(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_FLOAT: {
        float fv[ISPC_MAX_NVEC];
        constExpr->AsFloat(fv, forceVarying);
        return new ConstExpr(toType, fv, pos);
    }
    case AtomicType::TYPE_INT64: {
        int64_t iv[ISPC_MAX_NVEC];
        constExpr->AsInt64(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT64: {
        uint64_t uv[ISPC_MAX_NVEC];
        constExpr->AsUInt64(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_DOUBLE: {
        double dv[ISPC_MAX_NVEC];
        constExpr->AsDouble(dv, forceVarying);
        return new ConstExpr(toType, dv, pos);
    }
    default:
        FATAL("unimplemented");
    }
    return this;
}


void
TypeCastExpr::Print() const {
    printf("[%s] type cast (", GetType()->GetString().c_str());
    expr->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// ReferenceExpr

ReferenceExpr::ReferenceExpr(Expr *e, SourcePos p)
    : Expr(p) {
    expr = e;
}


llvm::Value *
ReferenceExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    return expr ? expr->GetLValue(ctx) : NULL;
}


Symbol *
ReferenceExpr::GetBaseSymbol() const {
    return expr ? expr->GetBaseSymbol() : NULL;
}


const Type *
ReferenceExpr::GetType() const {
    if (!expr) 
        return NULL;

    const Type *type = expr->GetType();
    if (!type) 
        return NULL;

    return new ReferenceType(type, false);
}


Expr *
ReferenceExpr::Optimize() {
    if (expr) 
        expr = expr->Optimize();
    if (expr == NULL)
        return NULL;

    return this;
}


Expr *
ReferenceExpr::TypeCheck() {
    if (expr != NULL) 
        expr = expr->TypeCheck();
    if (expr == NULL)
        return NULL;
    return this;
}


void
ReferenceExpr::Print() const {
    if (expr == NULL || GetType() == NULL)
        return;

    printf("[%s] &(", GetType()->GetString().c_str());
    expr->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// DereferenceExpr

DereferenceExpr::DereferenceExpr(Expr *e, SourcePos p)
    : Expr(p) {
    expr = e;
}


llvm::Value *
DereferenceExpr::GetValue(FunctionEmitContext *ctx) const {
    if (expr == NULL) 
        return NULL;
    llvm::Value *ptr = expr->GetValue(ctx);
    if (ptr == NULL)
        return NULL;
    const Type *type = GetType();
    if (type == NULL)
        return NULL;

    ctx->SetDebugPos(pos);
    return ctx->LoadInst(ptr, type, "reference_load");
}


llvm::Value *
DereferenceExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (expr == NULL) 
        return NULL;
    return expr->GetValue(ctx);
}


Symbol *
DereferenceExpr::GetBaseSymbol() const {
    return expr ? expr->GetBaseSymbol() : NULL;
}


const Type *
DereferenceExpr::GetType() const {
    return (expr && expr->GetType()) ? expr->GetType()->GetReferenceTarget() : 
        NULL;
}


Expr *
DereferenceExpr::TypeCheck() {
    if (expr != NULL)
        expr = expr->TypeCheck();
    if (expr == NULL)
        return NULL;
    return this;
}


Expr *
DereferenceExpr::Optimize() {
    if (expr != NULL) 
        expr = expr->Optimize();
    if (expr == NULL)
        return NULL;
    return this;
}


void
DereferenceExpr::Print() const {
    if (expr == NULL || GetType() == NULL)
        return;

    printf("[%s] *(", GetType()->GetString().c_str());
    expr->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// SymbolExpr

SymbolExpr::SymbolExpr(Symbol *s, SourcePos p) 
  : Expr(p) {
    symbol = s;
}


llvm::Value *
SymbolExpr::GetValue(FunctionEmitContext *ctx) const {
    // storagePtr may be NULL due to an earlier compilation error
    if (!symbol || !symbol->storagePtr)
        return NULL;
    ctx->SetDebugPos(pos);
    return ctx->LoadInst(symbol->storagePtr, GetType(), symbol->name.c_str());
}


llvm::Value *
SymbolExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (symbol == NULL)
        return NULL;
    ctx->SetDebugPos(pos);
    return symbol->storagePtr;
}


Symbol *
SymbolExpr::GetBaseSymbol() const {
    return symbol;
}


const Type *
SymbolExpr::GetType() const { 
    return symbol ? symbol->type : NULL;
}


Expr *
SymbolExpr::TypeCheck() {
    return this;
}


Expr *
SymbolExpr::Optimize() {
    if (symbol == NULL)
        return NULL;
    else if (symbol->constValue != NULL) {
        assert(GetType()->IsConstType());
        return symbol->constValue;
    }
    else
        return this;
}


void
SymbolExpr::Print() const {
    if (symbol == NULL || GetType() == NULL)
        return;

    printf("[%s] sym: (%s)", GetType()->GetString().c_str(), 
           symbol->name.c_str());
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// FunctionSymbolExpr

FunctionSymbolExpr::FunctionSymbolExpr(std::vector<Symbol *> *candidates,
                                       SourcePos p) 
  : Expr(p) {
    matchingFunc = NULL;
    candidateFunctions = candidates;
}


const Type *
FunctionSymbolExpr::GetType() const {
    return matchingFunc ? matchingFunc->type : NULL;
}


llvm::Value *
FunctionSymbolExpr::GetValue(FunctionEmitContext *ctx) const {
    assert("!should not call FunctionSymbolExpr::GetValue()");
    return NULL;
}


Symbol *
FunctionSymbolExpr::GetBaseSymbol() const {
    return matchingFunc;
}


Expr *
FunctionSymbolExpr::TypeCheck() {
    return this;
}


Expr *
FunctionSymbolExpr::Optimize() {
    return this;
}


void
FunctionSymbolExpr::Print() const {
    if (!matchingFunc || !GetType())
        return;

    printf("[%s] fun sym (%s)", GetType()->GetString().c_str(),
           matchingFunc->name.c_str());
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// SyncExpr

const Type *
SyncExpr::GetType() const {
    return AtomicType::Void;
}


llvm::Value *
SyncExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    std::vector<llvm::Value *> noArg;
    llvm::Function *fsync = m->module->getFunction("ISPCSync");
    if (fsync == NULL) {
        FATAL("Couldn't find ISPCSync declaration?!");
        return NULL;
    }

    return ctx->CallInst(fsync, noArg, "");
}


void
SyncExpr::Print() const {
    printf("sync");
    pos.Print();
}


Expr *
SyncExpr::TypeCheck() {
    return this;
}


Expr *
SyncExpr::Optimize() {
    return this;
}
