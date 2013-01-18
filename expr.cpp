/*
  Copyright (c) 2010-2012, Intel Corporation
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
#include "ast.h"
#include "type.h"
#include "sym.h"
#include "ctx.h"
#include "module.h"
#include "util.h"
#include "llvmutil.h"
#ifndef _MSC_VER
#include <inttypes.h>
#endif
#ifndef PRId64
#define PRId64 "lld"
#endif
#ifndef PRIu64
#define PRIu64 "llu"
#endif

#include <list>
#include <set>
#include <stdio.h>
#if defined(LLVM_3_1) || defined(LLVM_3_2)
  #include <llvm/Module.h>
  #include <llvm/Type.h>
  #include <llvm/Instructions.h>
  #include <llvm/Function.h>
  #include <llvm/DerivedTypes.h>
  #include <llvm/LLVMContext.h>
  #include <llvm/CallingConv.h>
#else
  #include <llvm/IR/Module.h>
  #include <llvm/IR/Type.h>
  #include <llvm/IR/Instructions.h>
  #include <llvm/IR/Function.h>
  #include <llvm/IR/DerivedTypes.h>
  #include <llvm/IR/LLVMContext.h>
  #include <llvm/IR/CallingConv.h>
#endif
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/Support/InstIterator.h>


/////////////////////////////////////////////////////////////////////////////////////
// Expr

llvm::Value *
Expr::GetLValue(FunctionEmitContext *ctx) const {
    // Expressions that can't provide an lvalue can just return NULL
    return NULL;
}


const Type *
Expr::GetLValueType() const {
    // This also only needs to be overrided by Exprs that implement the
    // GetLValue() method.
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
    // default implementation.
    return NULL;
}


#if 0
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
#endif

///////////////////////////////////////////////////////////////////////////

static Expr *
lArrayToPointer(Expr *expr) {
    AssertPos(expr->pos, expr && CastType<ArrayType>(expr->GetType()));

    Expr *zero = new ConstExpr(AtomicType::UniformInt32, 0, expr->pos);
    Expr *index = new IndexExpr(expr, zero, expr->pos);
    Expr *addr = new AddressOfExpr(index, expr->pos);
    addr = TypeCheck(addr);
    Assert(addr != NULL);
    addr = Optimize(addr);
    Assert(addr != NULL);
    return addr;
}


static bool
lIsAllIntZeros(Expr *expr) {
    const Type *type = expr->GetType();
    if (type == NULL || type->IsIntType() == false)
        return false;

    ConstExpr *ce = dynamic_cast<ConstExpr *>(expr);
    if (ce == NULL)
        return false;

    uint64_t vals[ISPC_MAX_NVEC];
    int count = ce->AsUInt64(vals);
    if (count == 1) 
        return (vals[0] == 0);
    else {
        for (int i = 0; i < count; ++i)
            if (vals[i] != 0)
                return false;
    }
    return true;
}


static bool
lDoTypeConv(const Type *fromType, const Type *toType, Expr **expr,
            bool failureOk, const char *errorMsgBase, SourcePos pos) {
    /* This function is way too long and complex.  Is type conversion stuff
       always this messy, or can this be cleaned up somehow? */
    AssertPos(pos, failureOk || errorMsgBase != NULL);

    if (toType == NULL || fromType == NULL)
        return false;

    // The types are equal; there's nothing to do
    if (Type::Equal(toType, fromType))
        return true;

    if (Type::Equal(fromType, AtomicType::Void)) {
        if (!failureOk)
            Error(pos, "Can't convert from \"void\" to \"%s\" for %s.",
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if (Type::Equal(toType, AtomicType::Void)) {
        if (!failureOk)
            Error(pos, "Can't convert type \"%s\" to \"void\" for %s.",
                  fromType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if (CastType<FunctionType>(fromType)) {
        if (CastType<PointerType>(toType) != NULL) {
            // Convert function type to pointer to function type
            if (expr != NULL) {
                Expr *aoe = new AddressOfExpr(*expr, (*expr)->pos);
                if (lDoTypeConv(aoe->GetType(), toType, &aoe, failureOk,
                                errorMsgBase, pos)) {
                    *expr = aoe;
                    return true;
                }
            }
            else
                return lDoTypeConv(PointerType::GetUniform(fromType), toType, NULL,
                                   failureOk, errorMsgBase, pos);
        }
        else {
            if (!failureOk)
                Error(pos, "Can't convert function type \"%s\" to \"%s\" for %s.",
                      fromType->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return false;
        }
    }
    if (CastType<FunctionType>(toType)) {
        if (!failureOk)
            Error(pos, "Can't convert from type \"%s\" to function type \"%s\" "
                  "for %s.", fromType->GetString().c_str(),
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if ((toType->GetSOAWidth() > 0 || fromType->GetSOAWidth() > 0) &&
        Type::Equal(toType->GetAsUniformType(), fromType->GetAsUniformType()) &&
        toType->GetSOAWidth() != fromType->GetSOAWidth()) {
        if (!failureOk)
            Error(pos, "Can't convert between types \"%s\" and \"%s\" with "
                  "different SOA widths for %s.", fromType->GetString().c_str(),
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    const ArrayType *toArrayType = CastType<ArrayType>(toType);
    const ArrayType *fromArrayType = CastType<ArrayType>(fromType);
    const VectorType *toVectorType = CastType<VectorType>(toType);
    const VectorType *fromVectorType = CastType<VectorType>(fromType);
    const StructType *toStructType = CastType<StructType>(toType);
    const StructType *fromStructType = CastType<StructType>(fromType);
    const EnumType *toEnumType = CastType<EnumType>(toType);
    const EnumType *fromEnumType = CastType<EnumType>(fromType);
    const AtomicType *toAtomicType = CastType<AtomicType>(toType);
    const AtomicType *fromAtomicType = CastType<AtomicType>(fromType);
    const PointerType *fromPointerType = CastType<PointerType>(fromType);
    const PointerType *toPointerType = CastType<PointerType>(toType);

    // Do this early, since for the case of a conversion like
    // "float foo[10]" -> "float * uniform foo", we have what's seemingly
    // a varying to uniform conversion (but not really)
    if (fromArrayType != NULL && toPointerType != NULL) {
        // can convert any array to a void pointer (both uniform and
        // varying).
        if (PointerType::IsVoidPointer(toPointerType))
            goto typecast_ok;

        // array to pointer to array element type
        const Type *eltType = fromArrayType->GetElementType();
        if (toPointerType->GetBaseType()->IsConstType())
            eltType = eltType->GetAsConstType();

        PointerType pt(eltType, toPointerType->GetVariability(),
                       toPointerType->IsConstType());
        if (Type::Equal(toPointerType, &pt))
            goto typecast_ok;
        else {
            if (!failureOk)
                Error(pos, "Can't convert from incompatible array type \"%s\" "
                      "to pointer type \"%s\" for %s.", 
                      fromType->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return false;
        }
    }

    if (toType->IsUniformType() && fromType->IsVaryingType()) {
        if (!failureOk)
            Error(pos, "Can't convert from type \"%s\" to type \"%s\" for %s.",
                  fromType->GetString().c_str(), toType->GetString().c_str(), 
                  errorMsgBase);
        return false;
    }

    if (fromPointerType != NULL) {
        if (CastType<AtomicType>(toType) != NULL &&
            toType->IsBoolType())
            // Allow implicit conversion of pointers to bools
            goto typecast_ok;

        if (toArrayType != NULL &&
            Type::Equal(fromType->GetBaseType(), toArrayType->GetElementType())) {
            // Can convert pointers to arrays of the same type
            goto typecast_ok;
        }
        if (toPointerType == NULL) {
            if (!failureOk)
                Error(pos, "Can't convert between from pointer type "
                      "\"%s\" to non-pointer type \"%s\" for %s.", 
                      fromType->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return false;
        }
        else if (fromPointerType->IsSlice() == true &&
                 toPointerType->IsSlice() == false) {
            if (!failureOk)
                Error(pos, "Can't convert from pointer to SOA type "
                      "\"%s\" to pointer to non-SOA type \"%s\" for %s.",
                      fromPointerType->GetAsNonSlice()->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return false;
        }
        else if (PointerType::IsVoidPointer(toPointerType)) {
            // any pointer type can be converted to a void *
            goto typecast_ok;
        }
        else if (PointerType::IsVoidPointer(fromPointerType) &&
                 expr != NULL &&
                 dynamic_cast<NullPointerExpr *>(*expr) != NULL) {
            // and a NULL convert to any other pointer type
            goto typecast_ok;
        }
        else if (!Type::Equal(fromPointerType->GetBaseType(), 
                              toPointerType->GetBaseType()) &&
                 !Type::Equal(fromPointerType->GetBaseType()->GetAsConstType(), 
                              toPointerType->GetBaseType())) {
            if (!failureOk)
                Error(pos, "Can't convert from pointer type \"%s\" to "
                      "incompatible pointer type \"%s\" for %s.",
                      fromPointerType->GetString().c_str(),
                      toPointerType->GetString().c_str(), errorMsgBase);
            return false;
        }

        if (toType->IsVaryingType() && fromType->IsUniformType())
            goto typecast_ok;

        if (toPointerType->IsSlice() == true &&
            fromPointerType->IsSlice() == false)
            goto typecast_ok;

        // Otherwise there's nothing to do
        return true;
    }

    if (toPointerType != NULL && fromAtomicType != NULL && 
        fromAtomicType->IsIntType() && expr != NULL &&
        lIsAllIntZeros(*expr)) {
        // We have a zero-valued integer expression, which can also be
        // treated as a NULL pointer that can be converted to any other
        // pointer type.
        Expr *npe = new NullPointerExpr(pos);
        if (lDoTypeConv(PointerType::Void, toType, &npe,
                        failureOk, errorMsgBase, pos)) {
            *expr = npe;
            return true;
        }
        return false;
    }

    // Need to check this early, since otherwise the [sic] "unbound"
    // variability of SOA struct types causes things to get messy if that
    // hasn't been detected...
    if (toStructType && fromStructType &&
        (toStructType->GetSOAWidth() != fromStructType->GetSOAWidth())) {
        if (!failureOk)
            Error(pos, "Can't convert between incompatible struct types \"%s\" "
                  "and \"%s\" for %s.", fromType->GetString().c_str(),
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    // Convert from type T -> const T; just return a TypeCast expr, which
    // can handle this
    if (Type::EqualIgnoringConst(toType, fromType) &&
        toType->IsConstType() == true &&
        fromType->IsConstType() == false)
        goto typecast_ok;
    
    if (CastType<ReferenceType>(fromType)) {
        if (CastType<ReferenceType>(toType)) {
            // Convert from a reference to a type to a const reference to a type;
            // this is handled by TypeCastExpr
            if (Type::Equal(toType->GetReferenceTarget(),
                            fromType->GetReferenceTarget()->GetAsConstType()))
                goto typecast_ok;

            const ArrayType *atFrom = 
                CastType<ArrayType>(fromType->GetReferenceTarget());
            const ArrayType *atTo = 
                CastType<ArrayType>(toType->GetReferenceTarget());

            if (atFrom != NULL && atTo != NULL && 
                Type::Equal(atFrom->GetElementType(), atTo->GetElementType())) {
                goto typecast_ok;
            }
            else {
                if (!failureOk)
                    Error(pos, "Can't convert between incompatible reference types \"%s\" "
                          "and \"%s\" for %s.", fromType->GetString().c_str(),
                          toType->GetString().c_str(), errorMsgBase);
                return false;
            }
        }
        else {
            // convert from a reference T -> T
            if (expr != NULL) {
                Expr *drExpr = new RefDerefExpr(*expr, pos);
                if (lDoTypeConv(drExpr->GetType(), toType, &drExpr, failureOk, 
                                errorMsgBase, pos) == true) {
                    *expr = drExpr;
                    return true;
                }
                return false;
            }
            else
                return lDoTypeConv(fromType->GetReferenceTarget(), toType, NULL,
                                   failureOk, errorMsgBase, pos);
        }
    }
    else if (CastType<ReferenceType>(toType)) {
        // T -> reference T
        if (expr != NULL) {
            Expr *rExpr = new ReferenceExpr(*expr, pos);
            if (lDoTypeConv(rExpr->GetType(), toType, &rExpr, failureOk,
                            errorMsgBase, pos) == true) {
                *expr = rExpr;
                return true;
            }
            return false;
        }
        else {
            ReferenceType rt(fromType);
            return lDoTypeConv(&rt, toType, NULL, failureOk, errorMsgBase, pos);
        }
    }
    else if (Type::Equal(toType, fromType->GetAsNonConstType()))
        // convert: const T -> T (as long as T isn't a reference)
        goto typecast_ok;

    fromType = fromType->GetReferenceTarget();
    toType = toType->GetReferenceTarget();
    if (toArrayType && fromArrayType) {
        if (Type::Equal(toArrayType->GetElementType(), 
                        fromArrayType->GetElementType())) {
            // the case of different element counts should have returned
            // successfully earlier, yes??
            AssertPos(pos, toArrayType->GetElementCount() != fromArrayType->GetElementCount());
            goto typecast_ok;
        }
        else if (Type::Equal(toArrayType->GetElementType(), 
                             fromArrayType->GetElementType()->GetAsConstType())) {
            // T[x] -> const T[x]
            goto typecast_ok;
        }
        else {
            if (!failureOk)
                Error(pos, "Array type \"%s\" can't be converted to type \"%s\" for %s.",
                      fromType->GetString().c_str(), toType->GetString().c_str(),
                      errorMsgBase);
            return false;
        }
    }

    if (toVectorType && fromVectorType) {
        // converting e.g. int<n> -> float<n>
        if (fromVectorType->GetElementCount() != toVectorType->GetElementCount()) {
            if (!failureOk)
                Error(pos, "Can't convert between differently sized vector types "
                      "\"%s\" -> \"%s\" for %s.", fromType->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return false;
        }
        goto typecast_ok;
    }

    if (toStructType && fromStructType) {
        if (!Type::Equal(toStructType->GetAsUniformType()->GetAsConstType(),
                         fromStructType->GetAsUniformType()->GetAsConstType())) {
            if (!failureOk)
                Error(pos, "Can't convert between different struct types "
                      "\"%s\" and \"%s\" for %s.", fromStructType->GetString().c_str(),
                      toStructType->GetString().c_str(), errorMsgBase);
            return false;
        }
        goto typecast_ok;
    }

    if (toEnumType != NULL && fromEnumType != NULL) {
        // No implicit conversions between different enum types
        if (!Type::EqualIgnoringConst(toEnumType->GetAsUniformType(),
                                     fromEnumType->GetAsUniformType())) {
            if (!failureOk)
                Error(pos, "Can't convert between different enum types "
                      "\"%s\" and \"%s\" for %s", fromEnumType->GetString().c_str(),
                      toEnumType->GetString().c_str(), errorMsgBase);
            return false;
        }
        goto typecast_ok;
    }

    // enum -> atomic (integer, generally...) is always ok
    if (fromEnumType != NULL) {
        AssertPos(pos, toAtomicType != NULL || toVectorType != NULL);
        goto typecast_ok;
    }

    // from here on out, the from type can only be atomic something or
    // other...
    if (fromAtomicType == NULL) {
        if (!failureOk)
            Error(pos, "Type conversion from \"%s\" to \"%s\" for %s is not "
                  "possible.", fromType->GetString().c_str(), 
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    // scalar -> short-vector conversions
    if (toVectorType != NULL &&
        (fromType->GetSOAWidth() == toType->GetSOAWidth()))
        goto typecast_ok;

    // ok, it better be a scalar->scalar conversion of some sort by now
    if (toAtomicType == NULL) {
        if (!failureOk)
            Error(pos, "Type conversion from \"%s\" to \"%s\" for %s is "
                  "not possible", fromType->GetString().c_str(), 
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if (fromType->GetSOAWidth() != toType->GetSOAWidth()) {
        if (!failureOk)
            Error(pos, "Can't convert between types \"%s\" and \"%s\" with "
                  "different SOA widths for %s.", fromType->GetString().c_str(),
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

 typecast_ok:
    if (expr != NULL)
        *expr = new TypeCastExpr(toType, *expr, pos);
    return true;
}


bool
CanConvertTypes(const Type *fromType, const Type *toType, 
                const char *errorMsgBase, SourcePos pos) {
    return lDoTypeConv(fromType, toType, NULL, errorMsgBase == NULL,
                       errorMsgBase, pos);
}


Expr *
TypeConvertExpr(Expr *expr, const Type *toType, const char *errorMsgBase) {
    if (expr == NULL)
        return NULL;

#if 0
    Debug(expr->pos, "type convert %s -> %s.", expr->GetType()->GetString().c_str(),
          toType->GetString().c_str());
#endif

    const Type *fromType = expr->GetType();
    Expr *e = expr;
    if (lDoTypeConv(fromType, toType, &e, false, errorMsgBase, 
                    expr->pos))
        return e;
    else
        return NULL;
}


bool
PossiblyResolveFunctionOverloads(Expr *expr, const Type *type) {
    FunctionSymbolExpr *fse = NULL;
    const FunctionType *funcType = NULL;
    if (CastType<PointerType>(type) != NULL &&
        (funcType = CastType<FunctionType>(type->GetBaseType())) &&
        (fse = dynamic_cast<FunctionSymbolExpr *>(expr)) != NULL) {
        // We're initializing a function pointer with a function symbol,
        // which in turn may represent an overloaded function.  So we need
        // to try to resolve the overload based on the type of the symbol
        // we're initializing here.
        std::vector<const Type *> paramTypes;
        for (int i = 0; i < funcType->GetNumParameters(); ++i)
            paramTypes.push_back(funcType->GetParameterType(i));

        if (fse->ResolveOverloads(expr->pos, paramTypes) == false)
            return false;
    }
    return true;
}



/** Utility routine that emits code to initialize a symbol given an
    initializer expression.

    @param ptr       Memory location of storage for the symbol's data
    @param symName   Name of symbol (used in error messages)
    @param symType   Type of variable being initialized
    @param initExpr  Expression for the initializer
    @param ctx       FunctionEmitContext to use for generating instructions
    @param pos       Source file position of the variable being initialized
*/
void
InitSymbol(llvm::Value *ptr, const Type *symType, Expr *initExpr, 
           FunctionEmitContext *ctx, SourcePos pos) {
    if (initExpr == NULL)
        // leave it uninitialized
        return;

    // See if we have a constant initializer a this point
    llvm::Constant *constValue = initExpr->GetConstant(symType);
    if (constValue != NULL) {
        // It'd be nice if we could just do a StoreInst(constValue, ptr)
        // at this point, but unfortunately that doesn't generate great
        // code (e.g. a bunch of scalar moves for a constant array.)  So
        // instead we'll make a constant static global that holds the
        // constant value and emit a memcpy to put its value into the
        // pointer we have.
        llvm::Type *llvmType = symType->LLVMType(g->ctx);
        if (llvmType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return;
        }

        if (Type::IsBasicType(symType))
            ctx->StoreInst(constValue, ptr);
        else {
            llvm::Value *constPtr = 
                new llvm::GlobalVariable(*m->module, llvmType, true /* const */, 
                                         llvm::GlobalValue::InternalLinkage,
                                         constValue, "const_initializer");
            llvm::Value *size = g->target.SizeOf(llvmType, 
                                                 ctx->GetCurrentBasicBlock());
            ctx->MemcpyInst(ptr, constPtr, size);
        }

        return;
    }

    // If the initializer is a straight up expression that isn't an
    // ExprList, then we'll see if we can type convert it to the type of
    // the variable.
    if (dynamic_cast<ExprList *>(initExpr) == NULL) {
        if (PossiblyResolveFunctionOverloads(initExpr, symType) == false)
            return;
        initExpr = TypeConvertExpr(initExpr, symType, "initializer");

        if (initExpr == NULL)
            return;

        llvm::Value *initializerValue = initExpr->GetValue(ctx);
        if (initializerValue != NULL)
            // Bingo; store the value in the variable's storage
            ctx->StoreInst(initializerValue, ptr);
        return;
    }

    // Atomic types and enums can be initialized with { ... } initializer
    // expressions if they have a single element (except for SOA types,
    // which are handled below).
    if (symType->IsSOAType() == false && Type::IsBasicType(symType)) {
        ExprList *elist = dynamic_cast<ExprList *>(initExpr);
        if (elist != NULL) {
            if (elist->exprs.size() == 1)
                InitSymbol(ptr, symType, elist->exprs[0], ctx, pos);
            else
                Error(initExpr->pos, "Expression list initializers with "
                      "multiple values can't be used with type \"%s\".", 
                      symType->GetString().c_str());
        }
        return;
    }

    const ReferenceType *rt = CastType<ReferenceType>(symType);
    if (rt) {
        if (!Type::Equal(initExpr->GetType(), rt)) {
            Error(initExpr->pos, "Initializer for reference type \"%s\" must have same "
                  "reference type itself. \"%s\" is incompatible.", 
                  rt->GetString().c_str(), initExpr->GetType()->GetString().c_str());
            return;
        }

        llvm::Value *initializerValue = initExpr->GetValue(ctx);
        if (initializerValue)
            ctx->StoreInst(initializerValue, ptr);
        return;
    }

    // Handle initiailizers for SOA types as well as for structs, arrays,
    // and vectors.
    const CollectionType *collectionType = CastType<CollectionType>(symType);
    if (collectionType != NULL || symType->IsSOAType()) {
        int nElements = collectionType ? collectionType->GetElementCount() :
            symType->GetSOAWidth();

        std::string name;
        if (CastType<StructType>(symType) != NULL)
            name = "struct";
        else if (CastType<ArrayType>(symType) != NULL) 
            name = "array";
        else if (CastType<VectorType>(symType) != NULL) 
            name = "vector";
        else if (symType->IsSOAType())
            name = symType->GetVariability().GetString();
        else
            FATAL("Unexpected CollectionType in InitSymbol()");

        // There are two cases for initializing these types; either a
        // single initializer may be provided (float foo[3] = 0;), in which
        // case all of the elements are initialized to the given value, or
        // an initializer list may be provided (float foo[3] = { 1,2,3 }),
        // in which case the elements are initialized with the
        // corresponding values.
        ExprList *exprList = dynamic_cast<ExprList *>(initExpr);
        if (exprList != NULL) {
            // The { ... } case; make sure we have the no more expressions
            // in the ExprList as we have struct members
            int nInits = exprList->exprs.size();
            if (nInits > nElements) {
                Error(initExpr->pos, "Initializer for %s type \"%s\" requires "
                      "no more than %d values; %d provided.", name.c_str(), 
                      symType->GetString().c_str(), nElements, nInits);
                return;
            }

            // Initialize each element with the corresponding value from
            // the ExprList
            for (int i = 0; i < nElements; ++i) {
                // For SOA types, the element type is the uniform variant
                // of the underlying type
                const Type *elementType = 
                    collectionType ? collectionType->GetElementType(i) : 
                                     symType->GetAsUniformType();
                if (elementType == NULL) {
                    AssertPos(pos, m->errorCount > 0);
                    return;
                }

                llvm::Value *ep;
                if (CastType<StructType>(symType) != NULL)
                    ep = ctx->AddElementOffset(ptr, i, NULL, "element");
                else
                    ep = ctx->GetElementPtrInst(ptr, LLVMInt32(0), LLVMInt32(i), 
                                                PointerType::GetUniform(elementType),
                                                "gep");

                if (i < nInits)
                    InitSymbol(ep, elementType, exprList->exprs[i], ctx, pos);
                else {
                    // If we don't have enough initializer values, initialize the
                    // rest as zero.
                    llvm::Type *llvmType = elementType->LLVMType(g->ctx);
                    if (llvmType == NULL) {
                        AssertPos(pos, m->errorCount > 0);
                        return;
                    }

                    llvm::Constant *zeroInit = llvm::Constant::getNullValue(llvmType);
                    ctx->StoreInst(zeroInit, ep);
                }
            }
        }
        else
            Error(initExpr->pos, "Can't assign type \"%s\" to \"%s\".",
                  initExpr->GetType()->GetString().c_str(),
                  collectionType->GetString().c_str());
        return;
    }

    FATAL("Unexpected Type in InitSymbol()");
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
    const VectorType *vt = CastType<VectorType>(type);
    if (vt != NULL)
        return new VectorType(boolBase, vt->GetElementCount());
    else {
        Assert(Type::IsBasicType(type));
        return boolBase;
    }
}

///////////////////////////////////////////////////////////////////////////
// UnaryExpr

static llvm::Constant *
lLLVMConstantValue(const Type *type, llvm::LLVMContext *ctx, double value) {
    const AtomicType *atomicType = CastType<AtomicType>(type);
    const EnumType *enumType = CastType<EnumType>(type);
    const VectorType *vectorType = CastType<VectorType>(type);
    const PointerType *pointerType = CastType<PointerType>(type);

    // This function is only called with, and only works for atomic, enum,
    // and vector types.
    Assert(atomicType != NULL || enumType != NULL || vectorType != NULL ||
           pointerType != NULL);

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
            Assert((double)i == value);
            return isUniform ? LLVMInt8(i) : LLVMInt8Vector(i);
        }
        case AtomicType::TYPE_UINT8: {
            unsigned int i = (unsigned int)value;
            return isUniform ? LLVMUInt8(i) : LLVMUInt8Vector(i);
        }
        case AtomicType::TYPE_INT16: {
            int i = (int)value;
            Assert((double)i == value);
            return isUniform ? LLVMInt16(i) : LLVMInt16Vector(i);
        }
        case AtomicType::TYPE_UINT16: {
            unsigned int i = (unsigned int)value;
            return isUniform ? LLVMUInt16(i) : LLVMUInt16Vector(i);
        }
        case AtomicType::TYPE_INT32: {
            int i = (int)value;
            Assert((double)i == value);
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
            Assert(value == (int64_t)i);
            return isUniform ? LLVMUInt64(i) : LLVMUInt64Vector(i);
        }
        case AtomicType::TYPE_INT64: {
            int64_t i = (int64_t)value;
            Assert((double)i == value);
            return isUniform ? LLVMInt64(i) : LLVMInt64Vector(i);
        }
        case AtomicType::TYPE_DOUBLE:
            return isUniform ? LLVMDouble(value) : LLVMDoubleVector(value);
        default:
            FATAL("logic error in lLLVMConstantValue");
            return NULL;
        }
    }
    else if (pointerType != NULL) {
        Assert(value == 0);
        if (pointerType->IsUniformType())
            return llvm::Constant::getNullValue(LLVMTypes::VoidPointerType);
        else
            return llvm::Constant::getNullValue(LLVMTypes::VoidPointerVectorType);
    }
    else {
        // For vector types, first get the LLVM constant for the basetype with
        // a recursive call to lLLVMConstantValue().
        const Type *baseType = vectorType->GetBaseType();
        llvm::Constant *constElement = lLLVMConstantValue(baseType, ctx, value);
        llvm::Type *llvmVectorType = vectorType->LLVMType(ctx);

        // Now create a constant version of the corresponding LLVM type that we
        // use to represent the VectorType.
        // FIXME: this is a little ugly in that the fact that ispc represents
        // uniform VectorTypes as LLVM VectorTypes and varying VectorTypes as
        // LLVM ArrayTypes leaks into the code here; it feels like this detail
        // should be better encapsulated?
        if (baseType->IsUniformType()) {
            llvm::VectorType *lvt = 
                llvm::dyn_cast<llvm::VectorType>(llvmVectorType);
            Assert(lvt != NULL);
            std::vector<llvm::Constant *> vals;
            for (unsigned int i = 0; i < lvt->getNumElements(); ++i)
                vals.push_back(constElement);
            return llvm::ConstantVector::get(vals);
        }
        else {
            llvm::ArrayType *lat = 
                llvm::dyn_cast<llvm::ArrayType>(llvmVectorType);
            Assert(lat != NULL);
            std::vector<llvm::Constant *> vals;
            for (unsigned int i = 0; i < lat->getNumElements(); ++i)
                vals.push_back(constElement);
            return llvm::ConstantArray::get(lat, vals);
        }
    }
}


static llvm::Value *
lMaskForSymbol(Symbol *baseSym, FunctionEmitContext *ctx) {
    if (baseSym == NULL)
        return ctx->GetFullMask();

    if (CastType<PointerType>(baseSym->type) != NULL ||
        CastType<ReferenceType>(baseSym->type) != NULL)
        // FIXME: for pointers, we really only want to do this for
        // dereferencing the pointer, not for things like pointer
        // arithmetic, when we may be able to use the internal mask,
        // depending on context...
        return ctx->GetFullMask();

    llvm::Value *mask = (baseSym->parentFunction == ctx->GetFunction() && 
                         baseSym->storageClass != SC_STATIC) ? 
        ctx->GetInternalMask() : ctx->GetFullMask();
    return mask;
}


/** Store the result of an assignment to the given location. 
 */
static void
lStoreAssignResult(llvm::Value *value, llvm::Value *ptr, const Type *valueType,
                   const Type *ptrType, FunctionEmitContext *ctx,
                   Symbol *baseSym) {
    Assert(baseSym == NULL ||
           baseSym->varyingCFDepth <= ctx->VaryingCFDepth());
    if (!g->opt.disableMaskedStoreToStore &&
        !g->opt.disableMaskAllOnOptimizations &&
        baseSym != NULL &&
        baseSym->varyingCFDepth == ctx->VaryingCFDepth() &&
        baseSym->storageClass != SC_STATIC &&
        CastType<ReferenceType>(baseSym->type) == NULL &&
        CastType<PointerType>(baseSym->type) == NULL) {
        // If the variable is declared at the same varying control flow
        // depth as where it's being assigned, then we don't need to do any
        // masking but can just do the assignment as if all the lanes were
        // known to be on.  While this may lead to random/garbage values
        // written into the lanes that are off, by definition they will
        // never be accessed, since those lanes aren't executing, and won't
        // be executing at this scope or any other one before the variable
        // goes out of scope.
        ctx->StoreInst(value, ptr, LLVMMaskAllOn, valueType, ptrType);
    }
    else {
        ctx->StoreInst(value, ptr, lMaskForSymbol(baseSym, ctx), valueType, ptrType);
    }
}


/** Utility routine to emit code to do a {pre,post}-{inc,dec}rement of the
    given expresion.
 */
static llvm::Value *
lEmitPrePostIncDec(UnaryExpr::Op op, Expr *expr, SourcePos pos,
                   FunctionEmitContext *ctx) {
    const Type *type = expr->GetType();
    if (type == NULL)
        return NULL;

    // Get both the lvalue and the rvalue of the given expression
    llvm::Value *lvalue = NULL, *rvalue = NULL;
    const Type *lvalueType = NULL;
    if (CastType<ReferenceType>(type) != NULL) {
        lvalueType = type;
        type = type->GetReferenceTarget();
        lvalue = expr->GetValue(ctx);

        Expr *deref = new RefDerefExpr(expr, expr->pos);
        rvalue = deref->GetValue(ctx);
    }
    else {
        lvalue = expr->GetLValue(ctx);
        lvalueType = expr->GetLValueType();
        rvalue = expr->GetValue(ctx);
    }

    if (lvalue == NULL) {
        // If we can't get a lvalue, then we have an error here 
        const char *prepost = (op == UnaryExpr::PreInc || 
                               op == UnaryExpr::PreDec) ? "pre" : "post";
        const char *incdec = (op == UnaryExpr::PreInc || 
                              op == UnaryExpr::PostInc) ? "increment" : "decrement";
        Error(pos, "Can't %s-%s non-lvalues.", prepost, incdec);
        return NULL;
    }

    // Emit code to do the appropriate addition/subtraction to the
    // expression's old value
    ctx->SetDebugPos(pos);
    llvm::Value *binop = NULL;
    int delta = (op == UnaryExpr::PreInc || op == UnaryExpr::PostInc) ? 1 : -1;

    std::string opName = rvalue->getName().str();
    if (op == UnaryExpr::PreInc || op == UnaryExpr::PostInc)
        opName += "_plus1";
    else
        opName += "_minus1";

    if (CastType<PointerType>(type) != NULL) {
        const Type *incType = type->IsUniformType() ? AtomicType::UniformInt32 :
            AtomicType::VaryingInt32;
        llvm::Constant *dval = lLLVMConstantValue(incType, g->ctx, delta);
        binop = ctx->GetElementPtrInst(rvalue, dval, type, opName.c_str());
    }
    else {
        llvm::Constant *dval = lLLVMConstantValue(type, g->ctx, delta);
        if (type->IsFloatType())
            binop = ctx->BinaryOperator(llvm::Instruction::FAdd, rvalue, 
                                        dval, opName.c_str());
        else
            binop = ctx->BinaryOperator(llvm::Instruction::Add, rvalue, 
                                        dval, opName.c_str());
    }

    // And store the result out to the lvalue
    Symbol *baseSym = expr->GetBaseSymbol();
    lStoreAssignResult(binop, lvalue, type, lvalueType, ctx, baseSym);

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
        return ctx->BinaryOperator(llvm::Instruction::FSub, zero, argVal,
                                   LLVMGetName(argVal, "_negate"));
    else {
        AssertPos(pos, type->IsIntType());
        return ctx->BinaryOperator(llvm::Instruction::Sub, zero, argVal,
                                   LLVMGetName(argVal, "_negate"));
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
        return ctx->NotOperator(argVal, LLVMGetName(argVal, "_logicalnot"));
    }
    case BitNot: {
        llvm::Value *argVal = expr->GetValue(ctx);
        return ctx->NotOperator(argVal, LLVMGetName(argVal, "_bitnot"));
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
    ConstExpr *constExpr = dynamic_cast<ConstExpr *>(expr);
    // If the operand isn't a constant, then we can't do any optimization
    // here...
    if (constExpr == NULL)
        return this;

    const Type *type = constExpr->GetType();
    bool isEnumType = CastType<EnumType>(type) != NULL;

    const Type *baseType = type->GetAsNonConstType()->GetAsUniformType();
    if (Type::Equal(baseType, AtomicType::UniformInt8) ||
        Type::Equal(baseType, AtomicType::UniformUInt8) ||
        Type::Equal(baseType, AtomicType::UniformInt16) ||
        Type::Equal(baseType, AtomicType::UniformUInt16) ||
        Type::Equal(baseType, AtomicType::UniformInt64) ||
        Type::Equal(baseType, AtomicType::UniformUInt64))
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
        // Since we currently only handle int32, floats, and doubles here,
        // it's safe to stuff whatever we have into a double, do the negate
        // as a double, and then return a ConstExpr with the same type as
        // the original...
        double v[ISPC_MAX_NVEC];
        int count = constExpr->AsDouble(v);
        for (int i = 0; i < count; ++i)
            v[i] = -v[i];
        return new ConstExpr(constExpr, v);
    }
    case BitNot: {
        if (Type::EqualIgnoringConst(type, AtomicType::UniformInt32) || 
            Type::EqualIgnoringConst(type, AtomicType::VaryingInt32)) {
            int32_t v[ISPC_MAX_NVEC];
            int count = constExpr->AsInt32(v);
            for (int i = 0; i < count; ++i)
                v[i] = ~v[i];
            return new ConstExpr(type, v, pos);
        }
        else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt32) || 
                 Type::EqualIgnoringConst(type, AtomicType::VaryingUInt32) ||
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
        AssertPos(pos, Type::EqualIgnoringConst(type, AtomicType::UniformBool) || 
               Type::EqualIgnoringConst(type, AtomicType::VaryingBool));
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
    const Type *type;
    if (expr == NULL || (type = expr->GetType()) == NULL)
        // something went wrong in type checking...
        return NULL;

    if (type->IsSOAType()) {
        Error(pos, "Can't apply unary operator to SOA type \"%s\".",
              type->GetString().c_str());
        return NULL;
    }

    if (op == PreInc || op == PreDec || op == PostInc || op == PostDec) {
        if (type->IsConstType()) {
            Error(pos, "Can't assign to type \"%s\" on left-hand side of "
                  "expression.", type->GetString().c_str());
            return NULL;
        }

        if (type->IsNumericType())
            return this;

        const PointerType *pt = CastType<PointerType>(type);
        if (pt == NULL) {
            Error(expr->pos, "Can only pre/post increment numeric and "
                  "pointer types, not \"%s\".", type->GetString().c_str());
            return NULL;
        }

        if (PointerType::IsVoidPointer(type)) {
            Error(expr->pos, "Illegal to pre/post increment \"%s\" type.",
                  type->GetString().c_str());
            return NULL;
        }
        if (CastType<UndefinedStructType>(pt->GetBaseType())) {
            Error(expr->pos, "Illegal to pre/post increment pointer to "
                  "undefined struct type \"%s\".", type->GetString().c_str());
            return NULL;
        }

        return this;
    }

    // don't do this for pre/post increment/decrement
    if (CastType<ReferenceType>(type)) {
        expr = new RefDerefExpr(expr, pos);
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
        expr = TypeConvertExpr(expr, boolType, "logical not");
        if (expr == NULL)
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


int
UnaryExpr::EstimateCost() const {
    if (dynamic_cast<ConstExpr *>(expr) != NULL)
        return 0;

    return COST_SIMPLE_ARITH_LOGIC_OP;
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
                 llvm::Value *arg1Val, bool isUnsigned,
                 FunctionEmitContext *ctx) {
    llvm::Instruction::BinaryOps inst;
    switch (op) {
    case BinaryExpr::Shl:    inst = llvm::Instruction::Shl;  break;
    case BinaryExpr::Shr:
        if (isUnsigned)
            inst = llvm::Instruction::LShr; 
        else
            inst = llvm::Instruction::AShr; 
        break; 
    case BinaryExpr::BitAnd: inst = llvm::Instruction::And;  break;
    case BinaryExpr::BitXor: inst = llvm::Instruction::Xor;  break;
    case BinaryExpr::BitOr:  inst = llvm::Instruction::Or;   break;
    default:
        FATAL("logic error in lEmitBinaryBitOp()");
        return NULL;
    }

    return ctx->BinaryOperator(inst, arg0Val, arg1Val, "bitop");
}


static llvm::Value *
lEmitBinaryPointerArith(BinaryExpr::Op op, llvm::Value *value0,
                        llvm::Value *value1, const Type *type0,
                        const Type *type1, FunctionEmitContext *ctx,
                        SourcePos pos) {
    const PointerType *ptrType = CastType<PointerType>(type0);

    switch (op) {
    case BinaryExpr::Add:
        // ptr + integer
        return ctx->GetElementPtrInst(value0, value1, ptrType, "ptrmath");
        break;
    case BinaryExpr::Sub: {
        if (CastType<PointerType>(type1) != NULL) {
            AssertPos(pos, Type::Equal(type0, type1));

            if (ptrType->IsSlice()) {
                llvm::Value *p0 = ctx->ExtractInst(value0, 0);
                llvm::Value *p1 = ctx->ExtractInst(value1, 0);
                const Type *majorType = ptrType->GetAsNonSlice();
                llvm::Value *majorDelta = 
                    lEmitBinaryPointerArith(op, p0, p1, majorType, majorType,
                                            ctx, pos);
               
                int soaWidth = ptrType->GetBaseType()->GetSOAWidth();
                AssertPos(pos, soaWidth > 0);
                llvm::Value *soaScale = LLVMIntAsType(soaWidth, 
                                                      majorDelta->getType());

                llvm::Value *majorScale =
                    ctx->BinaryOperator(llvm::Instruction::Mul, majorDelta,
                                        soaScale, "major_soa_scaled");

                llvm::Value *m0 = ctx->ExtractInst(value0, 1);
                llvm::Value *m1 = ctx->ExtractInst(value1, 1);
                llvm::Value *minorDelta =
                    ctx->BinaryOperator(llvm::Instruction::Sub, m0, m1,
                                        "minor_soa_delta");

                ctx->MatchIntegerTypes(&majorScale, &minorDelta);
                return ctx->BinaryOperator(llvm::Instruction::Add, majorScale,
                                           minorDelta, "soa_ptrdiff");
            }

            // ptr - ptr
            if (ptrType->IsUniformType()) {
                value0 = ctx->PtrToIntInst(value0);
                value1 = ctx->PtrToIntInst(value1);
            }

            // Compute the difference in bytes
            llvm::Value *delta = 
                ctx->BinaryOperator(llvm::Instruction::Sub, value0, value1,
                                    "ptr_diff");

            // Now divide by the size of the type that the pointer
            // points to in order to return the difference in elements.
            llvm::Type *llvmElementType = 
                ptrType->GetBaseType()->LLVMType(g->ctx);
            llvm::Value *size = g->target.SizeOf(llvmElementType, 
                                                 ctx->GetCurrentBasicBlock());
            if (ptrType->IsVaryingType())
                size = ctx->SmearUniform(size);

            if (g->target.is32Bit == false && 
                g->opt.force32BitAddressing == true) {
                // If we're doing 32-bit addressing math on a 64-bit
                // target, then trunc the delta down to a 32-bit value.
                // (Thus also matching what will be a 32-bit value
                // returned from SizeOf above.)
                if (ptrType->IsUniformType())
                    delta = ctx->TruncInst(delta, LLVMTypes::Int32Type,
                                           "trunc_ptr_delta");
                else
                    delta = ctx->TruncInst(delta, LLVMTypes::Int32VectorType,
                                           "trunc_ptr_delta");
            }

            // And now do the actual division
            return ctx->BinaryOperator(llvm::Instruction::SDiv, delta, size,
                                       "element_diff");
        }
        else {
            // ptr - integer
            llvm::Value *zero = lLLVMConstantValue(type1, g->ctx, 0.);
            llvm::Value *negOffset = 
                ctx->BinaryOperator(llvm::Instruction::Sub, zero, value1, 
                                    "negate");
            // Do a GEP as ptr + -integer
            return ctx->GetElementPtrInst(value0, negOffset, ptrType, 
                                          "ptrmath");
        }
    }
    default:
        FATAL("Logic error in lEmitBinaryArith() for pointer type case");
        return NULL;
    }

}

/** Utility routine to emit binary arithmetic operator based on the given
    BinaryExpr::Op.
*/
static llvm::Value *
lEmitBinaryArith(BinaryExpr::Op op, llvm::Value *value0, llvm::Value *value1,
                 const Type *type0, const Type *type1,
                 FunctionEmitContext *ctx, SourcePos pos) {
    const PointerType *ptrType = CastType<PointerType>(type0);

    if (ptrType != NULL)
        return lEmitBinaryPointerArith(op, value0, value1, type0, type1,
                                       ctx, pos);
    else {
        AssertPos(pos, Type::EqualIgnoringConst(type0, type1));

        llvm::Instruction::BinaryOps inst;
        bool isFloatOp = type0->IsFloatType();
        bool isUnsignedOp = type0->IsUnsignedType();

        const char *opName = NULL;
        switch (op) {
        case BinaryExpr::Add:
            opName = "add";
            inst = isFloatOp ? llvm::Instruction::FAdd : llvm::Instruction::Add;
            break;
        case BinaryExpr::Sub:
            opName = "sub";
            inst = isFloatOp ? llvm::Instruction::FSub : llvm::Instruction::Sub;
            break;
        case BinaryExpr::Mul:
            opName = "mul";
            inst = isFloatOp ? llvm::Instruction::FMul : llvm::Instruction::Mul;
            break;
        case BinaryExpr::Div:
            opName = "div";
            if (type0->IsVaryingType() && !isFloatOp)
                PerformanceWarning(pos, "Division with varying integer types is "
                                   "very inefficient."); 
            inst = isFloatOp ? llvm::Instruction::FDiv : 
                (isUnsignedOp ? llvm::Instruction::UDiv : llvm::Instruction::SDiv);
            break;
        case BinaryExpr::Mod:
            opName = "mod";
            if (type0->IsVaryingType() && !isFloatOp)
                PerformanceWarning(pos, "Modulus operator with varying types is "
                                   "very inefficient."); 
            inst = isFloatOp ? llvm::Instruction::FRem : 
                (isUnsignedOp ? llvm::Instruction::URem : llvm::Instruction::SRem);
            break;
        default:
            FATAL("Invalid op type passed to lEmitBinaryArith()");
            return NULL;
        }

        return ctx->BinaryOperator(inst, value0, value1, LLVMGetName(opName, value0, value1));
    }
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
    const char *opName = NULL;
    switch (op) {
    case BinaryExpr::Lt:
        opName = "less";
        pred = isFloatOp ? llvm::CmpInst::FCMP_ULT : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT);
        break;
    case BinaryExpr::Gt:
        opName = "greater";
        pred = isFloatOp ? llvm::CmpInst::FCMP_UGT : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT);
        break;
    case BinaryExpr::Le:
        opName = "lessequal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_ULE : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE);
        break;
    case BinaryExpr::Ge:
        opName = "greaterequal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_UGE : 
            (isUnsignedOp ? llvm::CmpInst::ICMP_UGE : llvm::CmpInst::ICMP_SGE);
        break;
    case BinaryExpr::Equal:
        opName = "equal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_UEQ : llvm::CmpInst::ICMP_EQ;
        break;
    case BinaryExpr::NotEqual:
        opName = "notequal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_UNE : llvm::CmpInst::ICMP_NE;
        break;
    default:
        FATAL("error in lEmitBinaryCmp()");
        return NULL;
    }

    llvm::Value *cmp = ctx->CmpInst(isFloatOp ? llvm::Instruction::FCmp : 
                                    llvm::Instruction::ICmp,
                                    pred, e0Val, e1Val, 
                                    LLVMGetName(opName, e0Val, e1Val));
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


/** Emit code for a && or || logical operator.  In particular, the code
    here handles "short-circuit" evaluation, where the second expression
    isn't evaluated if the value of the first one determines the value of
    the result. 
*/ 
llvm::Value *
lEmitLogicalOp(BinaryExpr::Op op, Expr *arg0, Expr *arg1,
               FunctionEmitContext *ctx, SourcePos pos) {

    const Type *type0 = arg0->GetType(), *type1 = arg1->GetType();
    if (type0 == NULL || type1 == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    // There is overhead (branches, etc.), to short-circuiting, so if the
    // right side of the expression is a) relatively simple, and b) can be
    // safely executed with an all-off execution mask, then we just
    // evaluate both sides and then the logical operator in that case.
    // FIXME: not sure what we should do about vector types here...
    bool shortCircuit = (EstimateCost(arg1) > PREDICATE_SAFE_IF_STATEMENT_COST ||
                         SafeToRunWithMaskAllOff(arg1) == false ||
                         CastType<VectorType>(type0) != NULL ||
                         CastType<VectorType>(type1) != NULL);
    if (shortCircuit == false) {
        // If one of the operands is uniform but the other is varying,
        // promote the uniform one to varying
        if (type0->IsUniformType() && type1->IsVaryingType()) {
            arg0 = TypeConvertExpr(arg0, AtomicType::VaryingBool, lOpString(op));
            AssertPos(pos, arg0 != NULL);
        }
        if (type1->IsUniformType() && type0->IsVaryingType()) {
            arg1 = TypeConvertExpr(arg1, AtomicType::VaryingBool, lOpString(op));
            AssertPos(pos, arg1 != NULL);
        }

        llvm::Value *value0 = arg0->GetValue(ctx);
        llvm::Value *value1 = arg1->GetValue(ctx);
        if (value0 == NULL || value1 == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        if (op == BinaryExpr::LogicalAnd)
            return ctx->BinaryOperator(llvm::Instruction::And, value0, value1,
                                       "logical_and");
        else {
            AssertPos(pos, op == BinaryExpr::LogicalOr);
            return ctx->BinaryOperator(llvm::Instruction::Or, value0, value1, 
                                       "logical_or");
        }
    }

    // Allocate temporary storage for the return value
    const Type *retType = Type::MoreGeneralType(type0, type1, pos, lOpString(op));
    llvm::Type *llvmRetType = retType->LLVMType(g->ctx);
    llvm::Value *retPtr = ctx->AllocaInst(llvmRetType, "logical_op_mem");

    llvm::BasicBlock *bbSkipEvalValue1 = ctx->CreateBasicBlock("skip_eval_1");
    llvm::BasicBlock *bbEvalValue1 = ctx->CreateBasicBlock("eval_1");
    llvm::BasicBlock *bbLogicalDone = ctx->CreateBasicBlock("logical_op_done");

    // Evaluate the first operand
    llvm::Value *value0 = arg0->GetValue(ctx);
    if (value0 == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    if (type0->IsUniformType()) {
        // Check to see if the value of the first operand is true or false
        llvm::Value *value0True = 
            ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ,
                         value0, LLVMTrue);

        if (op == BinaryExpr::LogicalOr) {
            // For ||, if value0 is true, then we skip evaluating value1
            // entirely.
            ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, value0True);

            // If value0 is true, the complete result is true (either
            // uniform or varying)
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            llvm::Value *trueValue = retType->IsUniformType() ? LLVMTrue :
                LLVMMaskAllOn;
            ctx->StoreInst(trueValue, retPtr);
            ctx->BranchInst(bbLogicalDone);
        }
        else {
            AssertPos(pos, op == BinaryExpr::LogicalAnd);

            // Conversely, for &&, if value0 is false, we skip evaluating
            // value1.
            ctx->BranchInst(bbEvalValue1, bbSkipEvalValue1, value0True);

            // In this case, the complete result is false (again, either a
            // uniform or varying false).
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            llvm::Value *falseValue = retType->IsUniformType() ? LLVMFalse :
                LLVMMaskAllOff;
            ctx->StoreInst(falseValue, retPtr);
            ctx->BranchInst(bbLogicalDone);
        }

        // Both || and && are in the same situation if the first operand's
        // value didn't resolve the final result: they need to evaluate the
        // value of the second operand, which in turn gives the value for
        // the full expression.
        ctx->SetCurrentBasicBlock(bbEvalValue1);
        if (type1->IsUniformType() && retType->IsVaryingType()) {
            arg1 = TypeConvertExpr(arg1, AtomicType::VaryingBool, "logical op");
            AssertPos(pos, arg1 != NULL);
        }

        llvm::Value *value1 = arg1->GetValue(ctx);
        if (value1 == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }
        ctx->StoreInst(value1, retPtr);
        ctx->BranchInst(bbLogicalDone);

        // In all cases, we end up at the bbLogicalDone basic block;
        // loading the value stored in retPtr in turn gives the overall
        // result.
        ctx->SetCurrentBasicBlock(bbLogicalDone);
        return ctx->LoadInst(retPtr);
    }
    else {
        // Otherwise, the first operand is varying...  Save the current
        // value of the mask so that we can restore it at the end.
        llvm::Value *oldMask = ctx->GetInternalMask();
        llvm::Value *oldFullMask = ctx->GetFullMask();

        // Convert the second operand to be varying as well, so that we can
        // perform logical vector ops with its value.
        if (type1->IsUniformType()) {
            arg1 = TypeConvertExpr(arg1, AtomicType::VaryingBool, "logical op");
            AssertPos(pos, arg1 != NULL);
            type1 = arg1->GetType();
        }

        if (op == BinaryExpr::LogicalOr) {
            // See if value0 is true for all currently executing
            // lanes--i.e. if (value0 & mask) == mask.  If so, we don't
            // need to evaluate the second operand of the expression.
            llvm::Value *value0AndMask = 
                ctx->BinaryOperator(llvm::Instruction::And, value0, 
                                    oldFullMask, "op&mask");
            llvm::Value *equalsMask =
                ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ,
                             value0AndMask, oldFullMask, "value0&mask==mask");
            equalsMask = ctx->I1VecToBoolVec(equalsMask);
            llvm::Value *allMatch = ctx->All(equalsMask);
            ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, allMatch);

            // value0 is true for all running lanes, so it can be used for
            // the final result
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            ctx->StoreInst(value0, retPtr);
            ctx->BranchInst(bbLogicalDone);

            // Otherwise, we need to valuate arg1. However, first we need
            // to set the execution mask to be (oldMask & ~a); in other
            // words, only execute the instances where value0 is false.
            // For the instances where value0 was true, we need to inhibit
            // execution.
            ctx->SetCurrentBasicBlock(bbEvalValue1);
            llvm::Value *not0 = ctx->NotOperator(value0);
            ctx->SetInternalMaskAnd(oldMask, not0);

            llvm::Value *value1 = arg1->GetValue(ctx);
            if (value1 == NULL) {
                AssertPos(pos, m->errorCount > 0);
                return NULL;
            }

            // We need to compute the result carefully, since vector
            // elements that were computed when the corresponding lane was
            // disabled have undefined values:
            // result = (value0 & old_mask) | (value1 & current_mask)
            llvm::Value *value1AndMask =
                ctx->BinaryOperator(llvm::Instruction::And, value1, 
                                    ctx->GetInternalMask(), "op&mask");
            llvm::Value *result =
                ctx->BinaryOperator(llvm::Instruction::Or, value0AndMask, 
                                    value1AndMask, "or_result");
            ctx->StoreInst(result, retPtr);
            ctx->BranchInst(bbLogicalDone);
        }
        else {
            AssertPos(pos, op == BinaryExpr::LogicalAnd);

            // If value0 is false for all currently running lanes, the
            // overall result must be false: this corresponds to checking
            // if (mask & ~value0) == mask.
            llvm::Value *notValue0 = ctx->NotOperator(value0, "not_value0");
            llvm::Value *notValue0AndMask = 
                ctx->BinaryOperator(llvm::Instruction::And, notValue0, 
                                    oldFullMask, "not_value0&mask");
            llvm::Value *equalsMask =
                ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ,
                             notValue0AndMask, oldFullMask, "not_value0&mask==mask");
            equalsMask = ctx->I1VecToBoolVec(equalsMask);
            llvm::Value *allMatch = ctx->All(equalsMask);
            ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, allMatch);

            // value0 was false for all running lanes, so use its value as
            // the overall result.
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            ctx->StoreInst(value0, retPtr);
            ctx->BranchInst(bbLogicalDone);

            // Otherwise we need to evaluate value1, but again with the
            // mask set to only be on for the lanes where value0 was true.
            // For the lanes where value0 was false, execution needs to be
            // disabled: mask = (mask & value0).
            ctx->SetCurrentBasicBlock(bbEvalValue1);
            ctx->SetInternalMaskAnd(oldMask, value0);

            llvm::Value *value1 = arg1->GetValue(ctx);
            if (value1 == NULL) {
                AssertPos(pos, m->errorCount > 0);
                return NULL;
            }

            // And as in the || case, we compute the overall result by
            // masking off the valid lanes before we AND them together:
            // result = (value0 & old_mask) & (value1 & current_mask)
            llvm::Value *value0AndMask = 
                ctx->BinaryOperator(llvm::Instruction::And, value0, 
                                    oldFullMask, "op&mask");
            llvm::Value *value1AndMask =
                ctx->BinaryOperator(llvm::Instruction::And, value1,
                                    ctx->GetInternalMask(), "value1&mask");
            llvm::Value *result =
                ctx->BinaryOperator(llvm::Instruction::And, value0AndMask, 
                                    value1AndMask, "or_result");
            ctx->StoreInst(result, retPtr);
            ctx->BranchInst(bbLogicalDone);
        }

        // And finally we always end up in bbLogicalDone, where we restore
        // the old mask and return the computed result
        ctx->SetCurrentBasicBlock(bbLogicalDone);
        ctx->SetInternalMask(oldMask);
        return ctx->LoadInst(retPtr);
    }
}


llvm::Value *
BinaryExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!arg0 || !arg1) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    // Handle these specially, since we want to short-circuit their evaluation...
    if (op == LogicalAnd || op == LogicalOr)
        return lEmitLogicalOp(op, arg0, arg1, ctx, pos);

    llvm::Value *value0 = arg0->GetValue(ctx);
    llvm::Value *value1 = arg1->GetValue(ctx);
    if (value0 == NULL || value1 == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    ctx->SetDebugPos(pos);

    switch (op) {
    case Add:
    case Sub:
    case Mul:
    case Div:
    case Mod:
        return lEmitBinaryArith(op, value0, value1, arg0->GetType(), arg1->GetType(),
                                ctx, pos);
    case Lt:
    case Gt:
    case Le:
    case Ge:
    case Equal:
    case NotEqual:
        return lEmitBinaryCmp(op, value0, value1, arg0->GetType(), ctx, pos);
    case Shl:
    case Shr:
    case BitAnd:
    case BitXor:
    case BitOr: {
        if (op == Shr && arg1->GetType()->IsVaryingType() && 
            dynamic_cast<ConstExpr *>(arg1) == NULL)
            PerformanceWarning(pos, "Shift right is extremely inefficient for "
                               "varying shift amounts.");
        return lEmitBinaryBitOp(op, value0, value1, 
                                arg0->GetType()->IsUnsignedType(), ctx);
    }
    case Comma:
        return value1;
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

    // If this hits, it means that our TypeCheck() method hasn't been
    // called before GetType() was called; adding two pointers is illegal
    // and will fail type checking and (int + ptr) should be canonicalized
    // into (ptr + int) by type checking.
    if (op == Add)
        AssertPos(pos, CastType<PointerType>(type1) == NULL);

    if (op == Comma)
        return arg1->GetType();

    if (CastType<PointerType>(type0) != NULL) {
        if (op == Add)
            // ptr + int -> ptr
            return type0;
        else if (op == Sub) {
            if (CastType<PointerType>(type1) != NULL) {
                // ptr - ptr -> ~ptrdiff_t
                const Type *diffType = (g->target.is32Bit || 
                                        g->opt.force32BitAddressing) ? 
                    AtomicType::UniformInt32 : AtomicType::UniformInt64;
                if (type0->IsVaryingType() || type1->IsVaryingType())
                    diffType = diffType->GetAsVaryingType();
                return diffType;
            }
            else
                // ptr - int -> ptr
                return type0;
        }

        // otherwise fall through for these...
        AssertPos(pos, op == Lt || op == Gt || op == Le || op == Ge ||
               op == Equal || op == NotEqual);
    }

    const Type *exprType = Type::MoreGeneralType(type0, type1, pos, lOpString(op));
    // I don't think that MoreGeneralType should be able to fail after the
    // checks done in BinaryExpr::TypeCheck().
    AssertPos(pos, exprType != NULL);

    switch (op) {
    case Add:
    case Sub:
    case Mul:
    case Div:
    case Mod:
        return exprType;
    case Lt:
    case Gt:
    case Le:
    case Ge:
    case Equal:
    case NotEqual:
    case LogicalAnd:
    case LogicalOr:
        return lMatchingBoolType(exprType);
    case Shl:
    case Shr:
        return type1->IsVaryingType() ? type0->GetAsVaryingType() : type0;
    case BitAnd:
    case BitXor:
    case BitOr:
        return exprType;
    case Comma:
        // handled above, so fall through here just in case
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

    const Type *rType = carg0->GetType()->IsUniformType() ? 
        AtomicType::UniformBool : AtomicType::VaryingBool;
    return new ConstExpr(rType, result, carg0->pos);
}


/** Constant fold binary arithmetic ops.
 */
template <typename T> static ConstExpr *
lConstFoldBinArithOp(BinaryExpr::Op op, const T *v0, const T *v1, ConstExpr *carg0,
                     SourcePos pos) {
    T result[ISPC_MAX_NVEC];
    int count = carg0->Count();

    switch (op) {
        FOLD_OP(BinaryExpr::Add, +);
        FOLD_OP(BinaryExpr::Sub, -);
        FOLD_OP(BinaryExpr::Mul, *);
    case BinaryExpr::Div:
        for (int i = 0; i < count; ++i) {
            if (v1[i] == 0) {
                Error(pos, "Division by zero encountered in expression.");
                result[i] = 0;
            }
            else
                result[i] = (v0[i] / v1[i]);
        }
        break;
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
    if (arg0 == NULL || arg1 == NULL)
        return NULL;

    ConstExpr *constArg0 = dynamic_cast<ConstExpr *>(arg0);
    ConstExpr *constArg1 = dynamic_cast<ConstExpr *>(arg1);

    if (g->opt.fastMath) {
        // optimizations related to division by floats..

        // transform x / const -> x * (1/const)
        if (op == Div && constArg1 != NULL) {
            const Type *type1 = constArg1->GetType();
            if (Type::EqualIgnoringConst(type1, AtomicType::UniformFloat) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingFloat)) {
                float inv[ISPC_MAX_NVEC];
                int count = constArg1->AsFloat(inv);
                for (int i = 0; i < count; ++i)
                    inv[i] = 1.f / inv[i];
                Expr *einv = new ConstExpr(type1, inv, constArg1->pos);
                Expr *e = new BinaryExpr(Mul, arg0, einv, pos);
                e = ::TypeCheck(e);
                if (e == NULL)
                    return NULL;
                return ::Optimize(e);
            }
        }

        // transform x / y -> x * rcp(y)
        if (op == Div) {
            const Type *type1 = arg1->GetType();
            if (Type::EqualIgnoringConst(type1, AtomicType::UniformFloat) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingFloat)) {
                // Get the symbol for the appropriate builtin
                std::vector<Symbol *> rcpFuns;
                m->symbolTable->LookupFunction("rcp", &rcpFuns);
                if (rcpFuns.size() > 0) {
                    Expr *rcpSymExpr = new FunctionSymbolExpr("rcp", rcpFuns, pos);
                    ExprList *args = new ExprList(arg1, arg1->pos);
                    Expr *rcpCall = new FunctionCallExpr(rcpSymExpr, args, 
                                                         arg1->pos);
                    rcpCall = ::TypeCheck(rcpCall);
                    if (rcpCall == NULL)
                        return NULL;
                    rcpCall = ::Optimize(rcpCall);
                    if (rcpCall == NULL)
                        return NULL;

                    Expr *ret = new BinaryExpr(Mul, arg0, rcpCall, pos);
                    ret = ::TypeCheck(ret);
                    if (ret == NULL)
                        return NULL;
                    return ::Optimize(ret);
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

    AssertPos(pos, Type::EqualIgnoringConst(arg0->GetType(), arg1->GetType()));
    const Type *type = arg0->GetType()->GetAsNonConstType();
    if (Type::Equal(type, AtomicType::UniformFloat) || 
        Type::Equal(type, AtomicType::VaryingFloat)) {
        float v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsFloat(v0);
        constArg1->AsFloat(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0, pos)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else 
            return this;
    }
    if (Type::Equal(type, AtomicType::UniformDouble) || 
        Type::Equal(type, AtomicType::VaryingDouble)) {
        double v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsDouble(v0);
        constArg1->AsDouble(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0, pos)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else 
            return this;
    }
    if (Type::Equal(type, AtomicType::UniformInt32) || 
        Type::Equal(type, AtomicType::VaryingInt32)) {
        int32_t v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsInt32(v0);
        constArg1->AsInt32(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0, pos)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinIntOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else
            return this;
    }
    else if (Type::Equal(type, AtomicType::UniformUInt32) || 
             Type::Equal(type, AtomicType::VaryingUInt32) ||
             CastType<EnumType>(type) != NULL) {
        uint32_t v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->AsUInt32(v0);
        constArg1->AsUInt32(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBinArithOp(op, v0, v1, constArg0, pos)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinIntOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else if ((ret = lConstFoldBinLogicalOp(op, v0, v1, constArg0)) != NULL)
            return ret;
        else
            return this;
    }
    else if (Type::Equal(type, AtomicType::UniformBool) || 
             Type::Equal(type, AtomicType::VaryingBool)) {
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
    if (arg0 == NULL || arg1 == NULL)
        return NULL;

    const Type *type0 = arg0->GetType(), *type1 = arg1->GetType();
    if (type0 == NULL || type1 == NULL)
        return NULL;

    // If either operand is a reference, dereference it before we move
    // forward
    if (CastType<ReferenceType>(type0) != NULL) {
        arg0 = new RefDerefExpr(arg0, arg0->pos);
        type0 = arg0->GetType();
        AssertPos(pos, type0 != NULL);
    }
    if (CastType<ReferenceType>(type1) != NULL) {
        arg1 = new RefDerefExpr(arg1, arg1->pos);
        type1 = arg1->GetType();
        AssertPos(pos, type1 != NULL);
    }

    // Convert arrays to pointers to their first elements
    if (CastType<ArrayType>(type0) != NULL) {
        arg0 = lArrayToPointer(arg0);
        type0 = arg0->GetType();
    }
    if (CastType<ArrayType>(type1) != NULL) {
        arg1 = lArrayToPointer(arg1);
        type1 = arg1->GetType();
    }

    // Prohibit binary operators with SOA types
    if (type0->GetSOAWidth() > 0) {
        Error(arg0->pos, "Illegal to use binary operator %s with SOA type "
              "\"%s\".", lOpString(op), type0->GetString().c_str());
        return NULL;
    }
    if (type1->GetSOAWidth() > 0) {
        Error(arg1->pos, "Illegal to use binary operator %s with SOA type "
              "\"%s\".", lOpString(op), type1->GetString().c_str());
        return NULL;
    }

    const PointerType *pt0 = CastType<PointerType>(type0);
    const PointerType *pt1 = CastType<PointerType>(type1);
    if (pt0 != NULL && pt1 != NULL && op == Sub) {
        // Pointer subtraction
        if (PointerType::IsVoidPointer(type0)) {
            Error(pos, "Illegal to perform pointer arithmetic "
                  "on \"%s\" type.", type0->GetString().c_str());
            return NULL;
        }
        if (PointerType::IsVoidPointer(type1)) {
            Error(pos, "Illegal to perform pointer arithmetic "
                  "on \"%s\" type.", type1->GetString().c_str());
            return NULL;
        }
        if (CastType<UndefinedStructType>(pt0->GetBaseType())) {
            Error(pos, "Illegal to perform pointer arithmetic "
                  "on undefined struct type \"%s\".", pt0->GetString().c_str());
            return NULL;
        }
        if (CastType<UndefinedStructType>(pt1->GetBaseType())) {
            Error(pos, "Illegal to perform pointer arithmetic "
                  "on undefined struct type \"%s\".", pt1->GetString().c_str());
            return NULL;
        }

        const Type *t = Type::MoreGeneralType(type0, type1, pos, "-");
        if (t == NULL)
            return NULL;

        arg0 = TypeConvertExpr(arg0, t, "pointer subtraction");
        arg1 = TypeConvertExpr(arg1, t, "pointer subtraction");
        if (arg0 == NULL || arg1 == NULL)
            return NULL;

        return this;
    }
    else if (((pt0 != NULL || pt1 != NULL) && op == Add) ||
             (pt0 != NULL && op == Sub)) {
        // Handle ptr + int, int + ptr, ptr - int
        if (pt0 != NULL && pt1 != NULL) {
            Error(pos, "Illegal to add two pointer types \"%s\" and \"%s\".",
                  pt0->GetString().c_str(), pt1->GetString().c_str());
            return NULL;
        }
        else if (pt1 != NULL) {
            // put in canonical order with the pointer as the first operand
            // for GetValue()
            std::swap(arg0, arg1);
            std::swap(type0, type1);
            std::swap(pt0, pt1);
        }

        AssertPos(pos, pt0 != NULL);

        if (PointerType::IsVoidPointer(pt0)) {
            Error(pos, "Illegal to perform pointer arithmetic "
                  "on \"%s\" type.", pt0->GetString().c_str());
            return NULL;
        }
        if (CastType<UndefinedStructType>(pt0->GetBaseType())) {
            Error(pos, "Illegal to perform pointer arithmetic "
                  "on undefined struct type \"%s\".", pt0->GetString().c_str());
            return NULL;
        }

        const Type *offsetType = g->target.is32Bit ? 
            AtomicType::UniformInt32 : AtomicType::UniformInt64;
        if (pt0->IsVaryingType())
            offsetType = offsetType->GetAsVaryingType();
        if (type1->IsVaryingType()) {
            arg0 = TypeConvertExpr(arg0, type0->GetAsVaryingType(), 
                                   "pointer addition");
            offsetType = offsetType->GetAsVaryingType();
            AssertPos(pos, arg0 != NULL);
        }

        arg1 = TypeConvertExpr(arg1, offsetType, lOpString(op));
        if (arg1 == NULL)
            return NULL;

        return this;
    }

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
                arg0 = TypeConvertExpr(arg0, type0->GetAsVaryingType(), 
                                       "shift operator");
                if (arg0 == NULL)
                    return NULL;
                type0 = arg0->GetType();
            }
            arg1 = TypeConvertExpr(arg1, type0, "shift operator");
            if (arg1 == NULL)
                return NULL;
        }
        else {
            const Type *promotedType = Type::MoreGeneralType(type0, type1, arg0->pos,
                                                             "binary bit op");
            if (promotedType == NULL)
                return NULL;

            arg0 = TypeConvertExpr(arg0, promotedType, "binary bit op");
            arg1 = TypeConvertExpr(arg1, promotedType, "binary bit op");
            if (arg0 == NULL || arg1 == NULL)
                return NULL;
        }
        return this;
    }
    case Add:
    case Sub:
    case Mul:
    case Div:
    case Mod: {
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

        const Type *promotedType = Type::MoreGeneralType(type0, type1, 
                                                         Union(arg0->pos, arg1->pos),
                                                         lOpString(op));
        if (promotedType == NULL)
            return NULL;

        arg0 = TypeConvertExpr(arg0, promotedType, lOpString(op));
        arg1 = TypeConvertExpr(arg1, promotedType, lOpString(op));
        if (arg0 == NULL || arg1 == NULL)
            return NULL;
        return this;
    }
    case Lt:
    case Gt:
    case Le:
    case Ge:
    case Equal:
    case NotEqual: {
        const PointerType *pt0 = CastType<PointerType>(type0);
        const PointerType *pt1 = CastType<PointerType>(type1);

        // Convert '0' in expressions where the other expression is a
        // pointer type to a NULL pointer.
        if (pt0 != NULL && lIsAllIntZeros(arg1)) {
            arg1 = new NullPointerExpr(pos);
            type1 = arg1->GetType();
            pt1 = CastType<PointerType>(type1);
        }
        else if (pt1 != NULL && lIsAllIntZeros(arg0)) {
            arg0 = new NullPointerExpr(pos);
            type0 = arg1->GetType();
            pt0 = CastType<PointerType>(type0);
        }

        if (pt0 == NULL && pt1 == NULL) {
            if (!type0->IsBoolType() && !type0->IsNumericType()) {
                Error(arg0->pos,
                      "First operand to operator \"%s\" is of "
                      "non-comparable type \"%s\".", lOpString(op), 
                      type0->GetString().c_str());
                return NULL;
            }
            if (!type1->IsBoolType() && !type1->IsNumericType()) {
                Error(arg1->pos,
                      "Second operand to operator \"%s\" is of "
                      "non-comparable type \"%s\".", lOpString(op), 
                      type1->GetString().c_str());
                return NULL;
            }
        }

        const Type *promotedType = 
            Type::MoreGeneralType(type0, type1, arg0->pos, lOpString(op));
        if (promotedType == NULL)
            return NULL;

        arg0 = TypeConvertExpr(arg0, promotedType, lOpString(op));
        arg1 = TypeConvertExpr(arg1, promotedType, lOpString(op));
        if (arg0 == NULL || arg1 == NULL)
            return NULL;
        return this;
    }
    case LogicalAnd:
    case LogicalOr: {
        // For now, we just type convert to boolean types, of the same
        // variability as the original types.  (When generating code, it's
        // useful to have preserved the uniform/varying distinction.)
        const AtomicType *boolType0 = type0->IsUniformType() ? 
            AtomicType::UniformBool : AtomicType::VaryingBool;
        const AtomicType *boolType1 = type1->IsUniformType() ? 
            AtomicType::UniformBool : AtomicType::VaryingBool;

        const Type *destType0 = NULL, *destType1 = NULL;
        const VectorType *vtype0 = CastType<VectorType>(type0);
        const VectorType *vtype1 = CastType<VectorType>(type1);
        if (vtype0 && vtype1) {
            int sz0 = vtype0->GetElementCount(), sz1 = vtype1->GetElementCount();
            if (sz0 != sz1) {
                Error(pos, "Can't do logical operation \"%s\" between vector types of "
                      "different sizes (%d vs. %d).", lOpString(op), sz0, sz1);
                return NULL;
            }
            destType0 = new VectorType(boolType0, sz0);
            destType1 = new VectorType(boolType1, sz1);
        }
        else if (vtype0 != NULL) {
            destType0 = new VectorType(boolType0, vtype0->GetElementCount());
            destType1 = new VectorType(boolType1, vtype0->GetElementCount());
        }
        else if (vtype1 != NULL) {
            destType0 = new VectorType(boolType0, vtype1->GetElementCount());
            destType1 = new VectorType(boolType1, vtype1->GetElementCount());
        }
        else {
            destType0 = boolType0;
            destType1 = boolType1;
        }

        arg0 = TypeConvertExpr(arg0, destType0, lOpString(op));
        arg1 = TypeConvertExpr(arg1, destType1, lOpString(op));
        if (arg0 == NULL || arg1 == NULL)
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


int
BinaryExpr::EstimateCost() const {
    if (dynamic_cast<ConstExpr *>(arg0) != NULL &&
        dynamic_cast<ConstExpr *>(arg1) != NULL)
        return 0;

    return (op == Div || op == Mod) ? COST_COMPLEX_ARITH_OP : 
                                      COST_SIMPLE_ARITH_LOGIC_OP;
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

static const char *
lOpString(AssignExpr::Op op) {
    switch (op) {
    case AssignExpr::Assign:    return "assignment operator";
    case AssignExpr::MulAssign: return "*=";
    case AssignExpr::DivAssign: return "/=";
    case AssignExpr::ModAssign: return "%%=";
    case AssignExpr::AddAssign: return "+=";
    case AssignExpr::SubAssign: return "-=";
    case AssignExpr::ShlAssign: return "<<=";
    case AssignExpr::ShrAssign: return ">>=";
    case AssignExpr::AndAssign: return "&=";
    case AssignExpr::XorAssign: return "^=";
    case AssignExpr::OrAssign:  return "|=";
    default:
        FATAL("Missing op in lOpstring");
        return "";
    }
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
    const Type *lvalueType = arg0->GetLValueType();
    const Type *resultType = arg0->GetType();
    if (lvalueType == NULL || resultType == NULL)
        return NULL;

    // Get the value on the right-hand side of the assignment+operation
    // operator and load the current value on the left-hand side.
    llvm::Value *rvalue = arg1->GetValue(ctx);
    llvm::Value *mask = lMaskForSymbol(baseSym, ctx);
    ctx->SetDebugPos(arg0->pos);
    llvm::Value *oldLHS = ctx->LoadInst(lv, mask, lvalueType);
    ctx->SetDebugPos(pos);

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
        newValue = lEmitBinaryArith(basicop, oldLHS, rvalue, type,
                                    arg1->GetType(), ctx, pos);
        break;
    case AssignExpr::ShlAssign:
    case AssignExpr::ShrAssign:
    case AssignExpr::AndAssign:
    case AssignExpr::XorAssign:
    case AssignExpr::OrAssign:
        newValue = lEmitBinaryBitOp(basicop, oldLHS, rvalue, 
                                    arg0->GetType()->IsUnsignedType(), ctx);
        break;
    default:
        FATAL("logic error in lEmitOpAssign");
        return NULL;
    }

    // And store the result back to the lvalue.
    ctx->SetDebugPos(arg0->pos);
    lStoreAssignResult(newValue, lv, resultType, lvalueType, ctx, baseSym);

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

    Symbol *baseSym = lvalue->GetBaseSymbol();

    switch (op) {
    case Assign: {
        llvm::Value *ptr = lvalue->GetLValue(ctx);
        if (ptr == NULL) {
            Error(lvalue->pos, "Left hand side of assignment expression can't "
                  "be assigned to.");
            return NULL;
        }
        const Type *ptrType = lvalue->GetLValueType();
        const Type *valueType = rvalue->GetType();
        if (ptrType == NULL || valueType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        llvm::Value *value = rvalue->GetValue(ctx);
        if (value == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        ctx->SetDebugPos(lvalue->pos);

        lStoreAssignResult(value, ptr, valueType, ptrType, ctx, baseSym);

        return value;
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
        AssertPos(pos, !CastType<ArrayType>(type) &&
               !CastType<StructType>(type));
        return lEmitOpAssign(op, lvalue, rvalue, type, baseSym, pos, ctx);
    }
    default:
        FATAL("logic error in AssignExpr::GetValue()");
        return NULL;
    }
}


Expr *
AssignExpr::Optimize() {
    if (lvalue == NULL || rvalue == NULL)
        return NULL;
    return this;
}


const Type *
AssignExpr::GetType() const {
    return lvalue ? lvalue->GetType() : NULL;
}


/** Recursively checks a structure type to see if it (or any struct type
    that it holds) has a const-qualified member. */
static bool
lCheckForConstStructMember(SourcePos pos, const StructType *structType,
                           const StructType *initialType) {
    for (int i = 0; i < structType->GetElementCount(); ++i) {
        const Type *t = structType->GetElementType(i);
        if (t->IsConstType()) {
            if (structType == initialType)
                Error(pos, "Illegal to assign to type \"%s\" due to element "
                      "\"%s\" with type \"%s\".", structType->GetString().c_str(),
                      structType->GetElementName(i).c_str(),
                      t->GetString().c_str());
            else
                Error(pos, "Illegal to assign to type \"%s\" in type \"%s\" "
                      "due to element \"%s\" with type \"%s\".", 
                      structType->GetString().c_str(),
                      initialType->GetString().c_str(), 
                      structType->GetElementName(i).c_str(),
                      t->GetString().c_str());
            return true;
        }

        const StructType *st = CastType<StructType>(t);
        if (st != NULL && lCheckForConstStructMember(pos, st, initialType))
            return true;
    }
    return false;
}


Expr *
AssignExpr::TypeCheck() {
    if (lvalue == NULL || rvalue == NULL) 
        return NULL;

    bool lvalueIsReference = 
        CastType<ReferenceType>(lvalue->GetType()) != NULL;
    if (lvalueIsReference)
        lvalue = new RefDerefExpr(lvalue, lvalue->pos);

    FunctionSymbolExpr *fse;
    if ((fse = dynamic_cast<FunctionSymbolExpr *>(rvalue)) != NULL) {
        // Special case to use the type of the LHS to resolve function
        // overloads when we're assigning a function pointer where the
        // function is overloaded.
        const Type *lvalueType = lvalue->GetType();
        const FunctionType *ftype;
        if (CastType<PointerType>(lvalueType) == NULL ||
            (ftype = CastType<FunctionType>(lvalueType->GetBaseType())) == NULL) {
            Error(lvalue->pos, "Can't assign function pointer to type \"%s\".",
                  lvalueType ? lvalueType->GetString().c_str() : "<unknown>");
            return NULL;
        }

        std::vector<const Type *> paramTypes;
        for (int i = 0; i < ftype->GetNumParameters(); ++i)
            paramTypes.push_back(ftype->GetParameterType(i));

        if (!fse->ResolveOverloads(rvalue->pos, paramTypes)) {
            Error(pos, "Unable to find overloaded function for function "
                  "pointer assignment.");
            return NULL;
        }
    }

    const Type *lhsType = lvalue->GetType();
    if (lhsType == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    if (lhsType->IsConstType()) {
        Error(lvalue->pos, "Can't assign to type \"%s\" on left-hand side of "
              "expression.", lhsType->GetString().c_str());
        return NULL;
    }

    if (CastType<PointerType>(lhsType) != NULL) {
        if (op == AddAssign || op == SubAssign) {
            if (PointerType::IsVoidPointer(lhsType)) {
                Error(pos, "Illegal to perform pointer arithmetic on \"%s\" "
                      "type.", lhsType->GetString().c_str());
                return NULL;
            }

            const Type *deltaType = g->target.is32Bit ? AtomicType::UniformInt32 :
                AtomicType::UniformInt64;
            if (lhsType->IsVaryingType())
                deltaType = deltaType->GetAsVaryingType();
            rvalue = TypeConvertExpr(rvalue, deltaType, lOpString(op));
        }
        else if (op == Assign)
            rvalue = TypeConvertExpr(rvalue, lhsType, "assignment");
        else {
            Error(lvalue->pos, "Assignment operator \"%s\" is illegal with pointer types.",
                  lOpString(op));
            return NULL;
        }
    }
    else if (CastType<ArrayType>(lhsType) != NULL) {
        Error(lvalue->pos, "Illegal to assign to array type \"%s\".",
              lhsType->GetString().c_str());
        return NULL;
    }
    else
        rvalue = TypeConvertExpr(rvalue, lhsType, lOpString(op));

    if (rvalue == NULL)
        return NULL;

    if (lhsType->IsFloatType() == true &&
        (op == ShlAssign || op == ShrAssign || op == AndAssign || 
         op == XorAssign || op == OrAssign)) {
            Error(pos, "Illegal to use %s operator with floating-point "
                  "operands.", lOpString(op));
            return NULL;
    }

    const StructType *st = CastType<StructType>(lhsType);
    if (st != NULL) {
        // Make sure we're not assigning to a struct that has a constant member
        if (lCheckForConstStructMember(pos, st, st))
            return NULL;
        
        if (op != Assign) {
            Error(lvalue->pos,  "Assignment operator \"%s\" is illegal with struct "
                  "type \"%s\".", lOpString(op), st->GetString().c_str());
            return NULL;
        }
    }

    return this;
}


int
AssignExpr::EstimateCost() const {
    if (op == Assign)
        return COST_ASSIGN;
    if (op == DivAssign || op == ModAssign)
        return COST_ASSIGN + COST_COMPLEX_ARITH_OP;
    else
        return COST_ASSIGN + COST_SIMPLE_ARITH_LOGIC_OP;
}


void
AssignExpr::Print() const {
    if (!lvalue || !rvalue || !GetType())
        return;

    printf("[%s] assign (", GetType()->GetString().c_str());
    lvalue->Print();
    printf(" %s ", lOpString(op));
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
    Assert(resultPtr->getType() ==
           PointerType::GetUniform(type)->LLVMType(g->ctx));
    ctx->StoreInst(expr1, resultPtr, test, type, PointerType::GetUniform(type));
    return ctx->LoadInst(resultPtr, "selectexpr_final");
}


static void
lEmitSelectExprCode(FunctionEmitContext *ctx, llvm::Value *testVal, 
                    llvm::Value *oldMask, llvm::Value *fullMask,
                    Expr *expr, llvm::Value *exprPtr) {
    llvm::BasicBlock *bbEval = ctx->CreateBasicBlock("select_eval_expr");
    llvm::BasicBlock *bbDone = ctx->CreateBasicBlock("select_done");

    // Check to see if the test was true for any of the currently executing
    // program instances.
    llvm::Value *testAndFullMask = 
        ctx->BinaryOperator(llvm::Instruction::And, testVal, fullMask, 
                            "test&mask");
    llvm::Value *anyOn = ctx->Any(testAndFullMask);
    ctx->BranchInst(bbEval, bbDone, anyOn);

    ctx->SetCurrentBasicBlock(bbEval);
    llvm::Value *testAndMask = 
        ctx->BinaryOperator(llvm::Instruction::And, testVal, oldMask, 
                            "test&mask");
    ctx->SetInternalMask(testAndMask);
    llvm::Value *exprVal = expr->GetValue(ctx);
    ctx->StoreInst(exprVal, exprPtr);
    ctx->BranchInst(bbDone);

    ctx->SetCurrentBasicBlock(bbDone);
}


llvm::Value *
SelectExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr1 || !expr2 || !test)
        return NULL;

    ctx->SetDebugPos(pos);

    const Type *testType = test->GetType()->GetAsNonConstType();
    // This should be taken care of during typechecking
    AssertPos(pos, Type::Equal(testType->GetBaseType(), AtomicType::UniformBool) ||
           Type::Equal(testType->GetBaseType(), AtomicType::VaryingBool));

    const Type *type = expr1->GetType();

    if (Type::Equal(testType, AtomicType::UniformBool)) {
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
    else if (CastType<VectorType>(testType) == NULL) {
        // the test is a varying bool type
        llvm::Value *testVal = test->GetValue(ctx);
        AssertPos(pos, testVal->getType() == LLVMTypes::MaskType);
        llvm::Value *oldMask = ctx->GetInternalMask();
        llvm::Value *fullMask = ctx->GetFullMask();

        // We don't want to incur the overhead for short-circuit evaluation
        // for expressions that are both computationally simple and safe to
        // run with an "all off" mask.
        bool shortCircuit1 =
            (::EstimateCost(expr1) > PREDICATE_SAFE_IF_STATEMENT_COST ||
             SafeToRunWithMaskAllOff(expr1) == false);
        bool shortCircuit2 =
            (::EstimateCost(expr2) > PREDICATE_SAFE_IF_STATEMENT_COST ||
             SafeToRunWithMaskAllOff(expr2) == false);

        Debug(expr1->pos, "%sshort circuiting evaluation for select expr",
              shortCircuit1 ? "" : "Not ");
        Debug(expr2->pos, "%sshort circuiting evaluation for select expr",
              shortCircuit2 ? "" : "Not ");

        // Temporary storage to store the values computed for each
        // expression, if any.  (These stay as uninitialized memory if we
        // short circuit around the corresponding expression.)
        llvm::Type *exprType = 
            expr1->GetType()->LLVMType(g->ctx);
        llvm::Value *expr1Ptr = ctx->AllocaInst(exprType);
        llvm::Value *expr2Ptr = ctx->AllocaInst(exprType);

        if (shortCircuit1)
            lEmitSelectExprCode(ctx, testVal, oldMask, fullMask, expr1, 
                                expr1Ptr);
        else {
            ctx->SetInternalMaskAnd(oldMask, testVal);
            llvm::Value *expr1Val = expr1->GetValue(ctx);
            ctx->StoreInst(expr1Val, expr1Ptr);
        }

        if (shortCircuit2) {
            llvm::Value *notTest = ctx->NotOperator(testVal);
            lEmitSelectExprCode(ctx, notTest, oldMask, fullMask, expr2, 
                                expr2Ptr);
        }
        else {
            ctx->SetInternalMaskAndNot(oldMask, testVal);
            llvm::Value *expr2Val = expr2->GetValue(ctx);
            ctx->StoreInst(expr2Val, expr2Ptr);
        }

        ctx->SetInternalMask(oldMask);
        llvm::Value *expr1Val = ctx->LoadInst(expr1Ptr);
        llvm::Value *expr2Val = ctx->LoadInst(expr2Ptr);
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
        const VectorType *vt = CastType<VectorType>(type);
        // Things that typechecking should have caught
        AssertPos(pos, vt != NULL);
        AssertPos(pos, CastType<VectorType>(testType) != NULL &&
               (CastType<VectorType>(testType)->GetElementCount() == 
                vt->GetElementCount()));

        // Do an element-wise select  
        llvm::Value *result = llvm::UndefValue::get(type->LLVMType(g->ctx));
        for (int i = 0; i < vt->GetElementCount(); ++i) {
            llvm::Value *ti = ctx->ExtractInst(testVal, i);
            llvm::Value *e1i = ctx->ExtractInst(expr1Val, i);
            llvm::Value *e2i = ctx->ExtractInst(expr2Val, i);
            llvm::Value *sel = NULL;
            if (testType->IsUniformType())
                sel = ctx->SelectInst(ti, e1i, e2i);
            else
                sel = lEmitVaryingSelect(ctx, ti, e1i, e2i, vt->GetElementType());
            result = ctx->InsertInst(result, sel, i);
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
    int testVecSize = CastType<VectorType>(testType) != NULL ?
        CastType<VectorType>(testType)->GetElementCount() : 0;
    int expr1VecSize = CastType<VectorType>(expr1Type) != NULL ?
        CastType<VectorType>(expr1Type)->GetElementCount() : 0;
    AssertPos(pos, !(testVecSize != 0 && expr1VecSize != 0 && testVecSize != expr1VecSize));
    
    int vectorSize = std::max(testVecSize, expr1VecSize);
    return Type::MoreGeneralType(expr1Type, expr2Type, Union(expr1->pos, expr2->pos),
                                 "select expression", becomesVarying, vectorSize);
}


Expr *
SelectExpr::Optimize() {
    if (test == NULL || expr1 == NULL || expr2 == NULL)
        return NULL;

    ConstExpr *constTest = dynamic_cast<ConstExpr *>(test);
    if (constTest == NULL)
        return this;

    // The test is a constant; see if we can resolve to one of the
    // expressions..
    bool bv[ISPC_MAX_NVEC];
    int count = constTest->AsBool(bv);
    if (count == 1)
        // Uniform test value; return the corresponding expression
        return (bv[0] == true) ? expr1 : expr2;
    else {
        // Varying test: see if all of the values are the same; if so, then
        // return the corresponding expression
        bool first = bv[0];
        bool mismatch = false;
        for (int i = 0; i < count; ++i)
            if (bv[i] != first) {
                mismatch = true;
                break;
            }
        if (mismatch == false)
            return (bv[0] == true) ? expr1 : expr2;

        // Last chance: see if the two expressions are constants; if so,
        // then we can do an element-wise selection based on the constant
        // condition..
        ConstExpr *constExpr1 = dynamic_cast<ConstExpr *>(expr1);
        ConstExpr *constExpr2 = dynamic_cast<ConstExpr *>(expr2);
        if (constExpr1 == NULL || constExpr2 == NULL)
            return this;

        AssertPos(pos, Type::Equal(constExpr1->GetType(), constExpr2->GetType()));
        const Type *exprType = constExpr1->GetType()->GetAsNonConstType();
        AssertPos(pos, exprType->IsVaryingType());

        // FIXME: it's annoying to have to have all of this replicated code.
        // FIXME: for completeness, it would also be nice to handle 8 and
        // 16 bit types...
        if (Type::Equal(exprType, AtomicType::VaryingInt32)) {
            int32_t v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
            int32_t result[ISPC_MAX_NVEC];
            constExpr1->AsInt32(v1);
            constExpr2->AsInt32(v2);
            for (int i = 0; i < count; ++i)
                result[i] = bv[i] ? v1[i] : v2[i];
            return new ConstExpr(exprType, result, pos);
        }
        if (Type::Equal(exprType, AtomicType::VaryingUInt32)) {
            uint32_t v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
            uint32_t result[ISPC_MAX_NVEC];
            constExpr1->AsUInt32(v1);
            constExpr2->AsUInt32(v2);
            for (int i = 0; i < count; ++i)
                result[i] = bv[i] ? v1[i] : v2[i];
            return new ConstExpr(exprType, result, pos);
        }
        else if (Type::Equal(exprType, AtomicType::VaryingInt64)) {
            int64_t v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
            int64_t result[ISPC_MAX_NVEC];
            constExpr1->AsInt64(v1);
            constExpr2->AsInt64(v2);
            for (int i = 0; i < count; ++i)
                result[i] = bv[i] ? v1[i] : v2[i];
            return new ConstExpr(exprType, result, pos);
        }
        else if (Type::Equal(exprType, AtomicType::VaryingUInt64)) {
            uint64_t v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
            uint64_t result[ISPC_MAX_NVEC];
            constExpr1->AsUInt64(v1);
            constExpr2->AsUInt64(v2);
            for (int i = 0; i < count; ++i)
                result[i] = bv[i] ? v1[i] : v2[i];
            return new ConstExpr(exprType, result, pos);
        }
        else if (Type::Equal(exprType, AtomicType::VaryingFloat)) {
            float v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
            float result[ISPC_MAX_NVEC];
            constExpr1->AsFloat(v1);
            constExpr2->AsFloat(v2);
            for (int i = 0; i < count; ++i)
                result[i] = bv[i] ? v1[i] : v2[i];
            return new ConstExpr(exprType, result, pos);
        }
        else if (Type::Equal(exprType, AtomicType::VaryingDouble)) {
            double v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
            double result[ISPC_MAX_NVEC];
            constExpr1->AsDouble(v1);
            constExpr2->AsDouble(v2);
            for (int i = 0; i < count; ++i)
                result[i] = bv[i] ? v1[i] : v2[i];
            return new ConstExpr(exprType, result, pos);
        }
        else if (Type::Equal(exprType, AtomicType::VaryingBool)) {
            bool v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
            bool result[ISPC_MAX_NVEC];
            constExpr1->AsBool(v1);
            constExpr2->AsBool(v2);
            for (int i = 0; i < count; ++i)
                result[i] = bv[i] ? v1[i] : v2[i];
            return new ConstExpr(exprType, result, pos);
        }

        return this;
    }
}


Expr *
SelectExpr::TypeCheck() {
    if (test == NULL || expr1 == NULL || expr2 == NULL)
        return NULL;

    const Type *type1 = expr1->GetType(), *type2 = expr2->GetType();
    if (!type1 || !type2)
        return NULL;

    if (const ArrayType *at1 = CastType<ArrayType>(type1)) {
        expr1 = TypeConvertExpr(expr1, PointerType::GetUniform(at1->GetBaseType()),
                                "select");
        if (expr1 == NULL)
            return NULL;
        type1 = expr1->GetType();
    }
    if (const ArrayType *at2 = CastType<ArrayType>(type2)) {
        expr2 = TypeConvertExpr(expr2, PointerType::GetUniform(at2->GetBaseType()),
                                "select");
        if (expr2 == NULL)
            return NULL;
        type2 = expr2->GetType();
    }

    const Type *testType = test->GetType();
    if (testType == NULL)
        return NULL;
    test = TypeConvertExpr(test, lMatchingBoolType(testType), "select");
    if (test == NULL)
        return NULL;
    testType = test->GetType();

    int testVecSize = CastType<VectorType>(testType) ?
        CastType<VectorType>(testType)->GetElementCount() : 0;
    const Type *promotedType = 
        Type::MoreGeneralType(type1, type2, Union(expr1->pos, expr2->pos),
                              "select expression", testType->IsVaryingType(), testVecSize);
    if (promotedType == NULL)
        return NULL;

    expr1 = TypeConvertExpr(expr1, promotedType, "select");
    expr2 = TypeConvertExpr(expr2, promotedType, "select");
    if (expr1 == NULL || expr2 == NULL)
        return NULL;

    return this;
}


int
SelectExpr::EstimateCost() const {
    return COST_SELECT;
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

FunctionCallExpr::FunctionCallExpr(Expr *f, ExprList *a, SourcePos p, 
                                   bool il, Expr *lce) 
    : Expr(p), isLaunch(il) {
    func = f;
    args = a;
    launchCountExpr = lce;
}


static const FunctionType *
lGetFunctionType(Expr *func) {
    if (func == NULL)
        return NULL;

    const Type *type = func->GetType();
    if (type == NULL)
        return NULL;

    const FunctionType *ftype = CastType<FunctionType>(type);
    if (ftype == NULL) {
        // Not a regular function symbol--is it a function pointer?
        if (CastType<PointerType>(type) != NULL)
            ftype = CastType<FunctionType>(type->GetBaseType());
    }
    return ftype;
}


llvm::Value *
FunctionCallExpr::GetValue(FunctionEmitContext *ctx) const {
    if (func == NULL || args == NULL)
        return NULL;

    ctx->SetDebugPos(pos);

    llvm::Value *callee = func->GetValue(ctx);

    if (callee == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    const FunctionType *ft = lGetFunctionType(func);
    AssertPos(pos, ft != NULL);
    bool isVoidFunc = Type::Equal(ft->GetReturnType(), AtomicType::Void);

    // Automatically convert function call args to references if needed.
    // FIXME: this should move to the TypeCheck() method... (but the
    // GetLValue call below needs a FunctionEmitContext, which is
    // problematic...)  
    std::vector<Expr *> callargs = args->exprs;
    bool err = false;

    // Specifically, this can happen if there's an error earlier during
    // overload resolution.
    if ((int)callargs.size() > ft->GetNumParameters()) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    for (unsigned int i = 0; i < callargs.size(); ++i) {
        Expr *argExpr = callargs[i];
        if (argExpr == NULL)
            continue;

        const Type *paramType = ft->GetParameterType(i); 

        const Type *argLValueType = argExpr->GetLValueType();
        if (argLValueType != NULL &&
            CastType<PointerType>(argLValueType) != NULL &&
            argLValueType->IsVaryingType() &&
            CastType<ReferenceType>(paramType) != NULL) {
            Error(argExpr->pos, "Illegal to pass a \"varying\" lvalue to a "
                  "reference parameter of type \"%s\".",
                  paramType->GetString().c_str());
            return NULL;
        }

        // Do whatever type conversion is needed
        argExpr = TypeConvertExpr(argExpr, paramType,
                                  "function call argument");
        if (argExpr == NULL)
            return NULL;
        callargs[i] = argExpr;
    }
    if (err)
        return NULL;

    // Fill in any default argument values needed.
    // FIXME: should we do this during type checking?
    for (int i = callargs.size(); i < ft->GetNumParameters(); ++i) {
        Expr *paramDefault = ft->GetParameterDefault(i);
        const Type *paramType = ft->GetParameterType(i);
        // FIXME: this type conv should happen when we create the function
        // type!
        Expr *d = TypeConvertExpr(paramDefault, paramType,
                                  "function call default argument");
        if (d == NULL)
            return NULL;
        callargs.push_back(d);
    }

    // Now evaluate the values of all of the parameters being passed.
    std::vector<llvm::Value *> argVals;
    for (unsigned int i = 0; i < callargs.size(); ++i) {
        Expr *argExpr = callargs[i];
        if (argExpr == NULL)
            // give up; we hit an error earlier
            return NULL;

        llvm::Value *argValue = argExpr->GetValue(ctx);
        if (argValue == NULL)
            // something went wrong in evaluating the argument's
            // expression, so give up on this
            return NULL;

        argVals.push_back(argValue);
    }


    llvm::Value *retVal = NULL;
    ctx->SetDebugPos(pos);
    if (ft->isTask) {
        AssertPos(pos, launchCountExpr != NULL);
        llvm::Value *launchCount = launchCountExpr->GetValue(ctx);
        if (launchCount != NULL)
            ctx->LaunchInst(callee, argVals, launchCount);
    }
    else
        retVal = ctx->CallInst(callee, ft, argVals, 
                               isVoidFunc ? "" : "calltmp");

    if (isVoidFunc)
        return NULL;
    else
        return retVal;
}


const Type *
FunctionCallExpr::GetType() const {
    const FunctionType *ftype = lGetFunctionType(func);
    return ftype ? ftype->GetReturnType() : NULL;
}


Expr *
FunctionCallExpr::Optimize() {
    if (func == NULL || args == NULL)
        return NULL;
    return this;
}


Expr *
FunctionCallExpr::TypeCheck() {
    if (func == NULL || args == NULL)
        return NULL;

    std::vector<const Type *> argTypes;
    std::vector<bool> argCouldBeNULL, argIsConstant;
    for (unsigned int i = 0; i < args->exprs.size(); ++i) {
        Expr *expr = args->exprs[i];

        if (expr == NULL)
            return NULL;
        const Type *t = expr->GetType();
        if (t == NULL)
            return NULL;

        argTypes.push_back(t);
        argCouldBeNULL.push_back(lIsAllIntZeros(expr) ||
                                 dynamic_cast<NullPointerExpr *>(expr));
        argIsConstant.push_back(dynamic_cast<ConstExpr *>(expr) ||
                                dynamic_cast<NullPointerExpr *>(expr));
    }

    FunctionSymbolExpr *fse = dynamic_cast<FunctionSymbolExpr *>(func);
    if (fse != NULL) {
        // Regular function call
        if (fse->ResolveOverloads(args->pos, argTypes, &argCouldBeNULL,
                                  &argIsConstant) == false)
            return NULL;

        func = ::TypeCheck(fse);
        if (func == NULL)
            return NULL;

        const FunctionType *ft = CastType<FunctionType>(func->GetType());
        if (ft == NULL) {
            const PointerType *pt = CastType<PointerType>(func->GetType());
            ft = (pt == NULL) ? NULL : 
                CastType<FunctionType>(pt->GetBaseType());
        }

        if (ft == NULL) {
            Error(pos, "Valid function name must be used for function call.");
            return NULL;
        }

        if (ft->isTask) {
            if (!isLaunch)
                Error(pos, "\"launch\" expression needed to call function "
                      "with \"task\" qualifier.");
            if (!launchCountExpr)
                return NULL;

            launchCountExpr = 
                TypeConvertExpr(launchCountExpr, AtomicType::UniformInt32,
                                "task launch count");
            if (launchCountExpr == NULL)
                return NULL;
        }
        else {
            if (isLaunch) {
                Error(pos, "\"launch\" expression illegal with non-\"task\"-"
                      "qualified function.");
                return NULL;
            }
            AssertPos(pos, launchCountExpr == NULL);
        }
    }
    else {
        // Call through a function pointer
        const Type *fptrType = func->GetType();
        if (fptrType == NULL)
            return NULL;

        // Make sure we do in fact have a function to call
        const FunctionType *funcType;
        if (CastType<PointerType>(fptrType) == NULL ||
            (funcType = CastType<FunctionType>(fptrType->GetBaseType())) == NULL) {
            Error(func->pos, "Must provide function name or function pointer for "
                  "function call expression.");
            return NULL;
        }
            
        // Make sure we don't have too many arguments for the function
        if ((int)argTypes.size() > funcType->GetNumParameters()) {
            Error(args->pos, "Too many parameter values provided in "
                  "function call (%d provided, %d expected).",
                  (int)argTypes.size(), funcType->GetNumParameters());
            return NULL;
        }
        // It's ok to have too few arguments, as long as the function's
        // default parameter values have started by the time we run out
        // of arguments
        if ((int)argTypes.size() < funcType->GetNumParameters() &&
            funcType->GetParameterDefault(argTypes.size()) == NULL) {
            Error(args->pos, "Too few parameter values provided in "
                  "function call (%d provided, %d expected).",
                  (int)argTypes.size(), funcType->GetNumParameters());
            return NULL;
        }

        // Now make sure they can all type convert to the corresponding
        // parameter types..
        for (int i = 0; i < (int)argTypes.size(); ++i) {
            if (i < funcType->GetNumParameters()) {
                // make sure it can type convert
                const Type *paramType = funcType->GetParameterType(i);
                if (CanConvertTypes(argTypes[i], paramType) == false &&
                    !(argCouldBeNULL[i] == true &&
                      CastType<PointerType>(paramType) != NULL)) {
                    Error(args->exprs[i]->pos, "Can't convert argument of "
                          "type \"%s\" to type \"%s\" for function call "
                          "argument.", argTypes[i]->GetString().c_str(),
                          paramType->GetString().c_str());
                    return NULL;
                }
            }
            else
                // Otherwise the parameter default saves us.  It should
                // be there for sure, given the check right above the
                // for loop.
                AssertPos(pos, funcType->GetParameterDefault(i) != NULL);
        }

        if (fptrType->IsVaryingType()) {
            const Type *retType = funcType->GetReturnType();
            if (Type::Equal(retType, AtomicType::Void) == false &&
                retType->IsUniformType()) {
                Error(pos, "Illegal to call a varying function pointer that "
                      "points to a function with a uniform return type \"%s\".",
                      funcType->GetReturnType()->GetString().c_str());
                return NULL;
            }
        }
    }

    if (func == NULL || args == NULL)
        return NULL;
    return this;
}


int
FunctionCallExpr::EstimateCost() const {
    if (isLaunch)
        return COST_TASK_LAUNCH;

    const Type *type = func->GetType();
    if (type == NULL)
        return 0;

    const PointerType *pt = CastType<PointerType>(type);
    if (pt != NULL)
        type = type->GetBaseType();
    const FunctionType *ftype = CastType<FunctionType>(type);

    if (ftype->costOverride > -1)
        return ftype->costOverride;

    if (pt != NULL)
        return pt->IsUniformType() ? COST_FUNPTR_UNIFORM : COST_FUNPTR_VARYING;
    else
        return COST_FUNCALL;
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
    return this;
}


ExprList *
ExprList::TypeCheck() {
    return this;
}


llvm::Constant *
ExprList::GetConstant(const Type *type) const {
    if (exprs.size() == 1 &&
        (CastType<AtomicType>(type) != NULL ||
         CastType<EnumType>(type) != NULL ||
         CastType<PointerType>(type) != NULL))
        return exprs[0]->GetConstant(type);

    const CollectionType *collectionType = CastType<CollectionType>(type);
    if (collectionType == NULL)
        return NULL;

    std::string name;
    if (CastType<StructType>(type) != NULL)
        name = "struct";
    else if (CastType<ArrayType>(type) != NULL) 
        name = "array";
    else if (CastType<VectorType>(type) != NULL) 
        name = "vector";
    else 
        FATAL("Unexpected CollectionType in ExprList::GetConstant()");

    if ((int)exprs.size() > collectionType->GetElementCount()) {
        Error(pos, "Initializer list for %s \"%s\" must have no more than %d "
              "elements (has %d).", name.c_str(), 
              collectionType->GetString().c_str(),
              collectionType->GetElementCount(), (int)exprs.size());
        return NULL;
    }

    std::vector<llvm::Constant *> cv;
    for (unsigned int i = 0; i < exprs.size(); ++i) {
        if (exprs[i] == NULL)
            return NULL;
        const Type *elementType = collectionType->GetElementType(i);

        Expr *expr = exprs[i];
        if (dynamic_cast<ExprList *>(expr) == NULL) {
            // If there's a simple type conversion from the type of this
            // expression to the type we need, then let the regular type
            // conversion machinery handle it.
            expr = TypeConvertExpr(exprs[i], elementType, "initializer list");
            if (expr == NULL) {
                AssertPos(pos, m->errorCount > 0);
                return NULL;
            }
            // Re-establish const-ness if possible
            expr = ::Optimize(expr);
        }

        llvm::Constant *c = expr->GetConstant(elementType);
        if (c == NULL)
            // If this list element couldn't convert to the right constant
            // type for the corresponding collection member, then give up.
            return NULL;
        cv.push_back(c);
    }

    // If there are too few, then treat missing ones as if they were zero
    for (int i = (int)exprs.size(); i < collectionType->GetElementCount(); ++i) {
        const Type *elementType = collectionType->GetElementType(i);
        if (elementType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        llvm::Type *llvmType = elementType->LLVMType(g->ctx);
        if (llvmType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        llvm::Constant *c = llvm::Constant::getNullValue(llvmType);
        cv.push_back(c);
    }

    if (CastType<StructType>(type) != NULL) {
        llvm::StructType *llvmStructType =
            llvm::dyn_cast<llvm::StructType>(collectionType->LLVMType(g->ctx));
        AssertPos(pos, llvmStructType != NULL);
        return llvm::ConstantStruct::get(llvmStructType, cv);
    }
    else {
        llvm::Type *lt = type->LLVMType(g->ctx);
        llvm::ArrayType *lat = 
            llvm::dyn_cast<llvm::ArrayType>(lt);
        if (lat != NULL)
            return llvm::ConstantArray::get(lat, cv);
        else {
            // uniform short vector type
            AssertPos(pos, type->IsUniformType() &&
                   CastType<VectorType>(type) != NULL);

            llvm::VectorType *lvt = 
                llvm::dyn_cast<llvm::VectorType>(lt);
            AssertPos(pos, lvt != NULL);

            // Uniform short vectors are stored as vectors of length
            // rounded up to the native vector width.  So we add additional
            // undef values here until we get the right size.
            int vectorWidth = g->target.nativeVectorWidth;
            const VectorType *vt = CastType<VectorType>(type);
            const AtomicType *bt = vt->GetElementType();

            if (Type::Equal(bt->GetAsUniformType(), AtomicType::UniformInt64) ||
                Type::Equal(bt->GetAsUniformType(), AtomicType::UniformUInt64) ||
                Type::Equal(bt->GetAsUniformType(), AtomicType::UniformDouble)) {
                // target.nativeVectorWidth should be in terms of 32-bit
                // values, so for the 64-bit guys, it takes half as many of
                // them to fill the native width
                vectorWidth /= 2;
            }
            
            while ((cv.size() % vectorWidth) != 0) {
                cv.push_back(llvm::UndefValue::get(lvt->getElementType()));
            }

            return llvm::ConstantVector::get(cv);
        }
    }
    return NULL;
}


int
ExprList::EstimateCost() const {
    return 0;
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
    baseExpr = a;
    index = i;
    type = lvalueType = NULL;
}


/** When computing pointer values, we need to apply a per-lane offset when
    we have a varying pointer that is itself indexing into varying data.
    Consdier the following ispc code:

    uniform float u[] = ...;
    float v[] = ...;
    int index = ...;
    float a = u[index];
    float b = v[index];

    To compute the varying pointer that holds the addresses to load from
    for u[index], we basically just need to multiply index element-wise by
    sizeof(float) before doing the memory load.  For v[index], we need to
    do the same scaling but also need to add per-lane offsets <0,
    sizeof(float), 2*sizeof(float), ...> so that the i'th lane loads the
    i'th of the varying values at its index value.  

    This function handles figuring out when this additional offset is
    needed and then incorporates it in the varying pointer value.
 */ 
static llvm::Value *
lAddVaryingOffsetsIfNeeded(FunctionEmitContext *ctx, llvm::Value *ptr, 
                           const Type *ptrRefType) {
    if (CastType<ReferenceType>(ptrRefType) != NULL)
        // References are uniform pointers, so no offsetting is needed
        return ptr;

    const PointerType *ptrType = CastType<PointerType>(ptrRefType);
    Assert(ptrType != NULL);
    if (ptrType->IsUniformType() || ptrType->IsSlice())
        return ptr;

    const Type *baseType = ptrType->GetBaseType();
    if (baseType->IsVaryingType() == false)
        return ptr;

    // must be indexing into varying atomic, enum, or pointer types
    if (Type::IsBasicType(baseType) == false)
        return ptr;

    // Onward: compute the per lane offsets.
    llvm::Value *varyingOffsets = 
        llvm::UndefValue::get(LLVMTypes::Int32VectorType);
    for (int i = 0; i < g->target.vectorWidth; ++i)
        varyingOffsets = ctx->InsertInst(varyingOffsets, LLVMInt32(i), i,
                                         "varying_delta");

    // And finally add the per-lane offsets.  Note that we lie to the GEP
    // call and tell it that the pointers are to uniform elements and not
    // varying elements, so that the offsets in terms of (0,1,2,...) will
    // end up turning into the correct step in bytes...
    const Type *uniformElementType = baseType->GetAsUniformType();
    const Type *ptrUnifType = PointerType::GetVarying(uniformElementType);
    return ctx->GetElementPtrInst(ptr, varyingOffsets, ptrUnifType);
}


/** Check to see if the given type is an array of or pointer to a varying
    struct type that in turn has a member with bound 'uniform' variability.
    Issue an error and return true if such a member is found.
 */
static bool
lVaryingStructHasUniformMember(const Type *type, SourcePos pos) {
    if (CastType<VectorType>(type) != NULL ||
        CastType<ReferenceType>(type) != NULL)
        return false;

    const StructType *st = CastType<StructType>(type);
    if (st == NULL) {
        const ArrayType *at = CastType<ArrayType>(type);
        if (at != NULL)
            st = CastType<StructType>(at->GetElementType());
        else {
            const PointerType *pt = CastType<PointerType>(type);
            if (pt == NULL)
                return false;

            st = CastType<StructType>(pt->GetBaseType());
        }

        if (st == NULL)
            return false;
    }

    if (st->IsVaryingType() == false)
        return false;

    for (int i = 0; i < st->GetElementCount(); ++i) {
        const Type *eltType = st->GetElementType(i);
        if (eltType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            continue;
        }

        if (CastType<StructType>(eltType) != NULL) {
            // We know that the enclosing struct is varying at this point,
            // so push that down to the enclosed struct before makign the
            // recursive call.
            eltType = eltType->GetAsVaryingType();
            if (lVaryingStructHasUniformMember(eltType, pos))
                return true;
        }
        else if (eltType->IsUniformType()) {
            Error(pos, "Gather operation is impossible due to the presence of "
                  "struct member \"%s\" with uniform type \"%s\" in the "
                  "varying struct type \"%s\".",
                  st->GetElementName(i).c_str(), eltType->GetString().c_str(),
                  st->GetString().c_str());
            return true;
        }
    }

    return false;
}


llvm::Value *
IndexExpr::GetValue(FunctionEmitContext *ctx) const {
    const Type *indexType, *returnType;
    if (baseExpr == NULL || index == NULL || 
        ((indexType = index->GetType()) == NULL) ||
        ((returnType = GetType()) == NULL)) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    // If this is going to be a gather, make sure that the varying return
    // type can represent the result (i.e. that we don't have a bound
    // 'uniform' member in a varying struct...)
    if (indexType->IsVaryingType() && 
        lVaryingStructHasUniformMember(returnType, pos))
        return NULL;

    ctx->SetDebugPos(pos);

    llvm::Value *ptr = GetLValue(ctx);
    llvm::Value *mask = NULL;
    const Type *lvType = GetLValueType();
    if (ptr == NULL) {
        // We may be indexing into a temporary that hasn't hit memory, so
        // get the full value and stuff it into temporary alloca'd space so
        // that we can index from there...
        const Type *baseExprType = baseExpr->GetType();
        llvm::Value *val = baseExpr->GetValue(ctx);
        if (baseExprType == NULL || val == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }
        ctx->SetDebugPos(pos);
        llvm::Value *tmpPtr = ctx->AllocaInst(baseExprType->LLVMType(g->ctx), 
                                              "array_tmp");
        ctx->StoreInst(val, tmpPtr);

        // Get a pointer type to the underlying elements
        const SequentialType *st = CastType<SequentialType>(baseExprType);
        if (st == NULL) {
            Assert(m->errorCount > 0);
            return NULL;
        }
        lvType = PointerType::GetUniform(st->GetElementType());

        // And do the indexing calculation into the temporary array in memory
        ptr = ctx->GetElementPtrInst(tmpPtr, LLVMInt32(0), index->GetValue(ctx), 
                                     PointerType::GetUniform(baseExprType));
        ptr = lAddVaryingOffsetsIfNeeded(ctx, ptr, lvType);

        mask = LLVMMaskAllOn;
    }
    else {
        Symbol *baseSym = GetBaseSymbol();
        AssertPos(pos, baseSym != NULL);
        mask = lMaskForSymbol(baseSym, ctx);
    }

    ctx->SetDebugPos(pos);
    return ctx->LoadInst(ptr, mask, lvType);
}


const Type *
IndexExpr::GetType() const {
    if (type != NULL)
        return type;

    const Type *baseExprType, *indexType;
    if (!baseExpr || !index || 
        ((baseExprType = baseExpr->GetType()) == NULL) ||
        ((indexType = index->GetType()) == NULL))
        return NULL;

    const Type *elementType = NULL;
    const PointerType *pointerType = CastType<PointerType>(baseExprType);
    if (pointerType != NULL)
        // ptr[index] -> type that the pointer points to
        elementType = pointerType->GetBaseType();
    else {
        // sequential type[index] -> element type of the sequential type
        const SequentialType *sequentialType = 
            CastType<SequentialType>(baseExprType->GetReferenceTarget());
        // Typechecking should have caught this...
        AssertPos(pos, sequentialType != NULL);
        elementType = sequentialType->GetElementType();
    }

    // If we're indexing into a sequence of SOA types, the result type is
    // actually the underlying type, as a uniform or varying.  Get the
    // uniform variant of it for starters, then below we'll make it varying
    // if the index is varying.
    // (If we ever provide a way to index into SOA types and get an entire
    // SOA'd struct out of the array, then we won't want to do this in that
    // case..)
    if (elementType->IsSOAType())
        elementType = elementType->GetAsUniformType();

    // If either the index is varying or we're indexing into a varying
    // pointer, then the result type is the varying variant of the indexed
    // type.
    if (indexType->IsUniformType() &&
        (pointerType == NULL || pointerType->IsUniformType()))
        type = elementType;
    else
        type = elementType->GetAsVaryingType();

    return type;
}


Symbol *
IndexExpr::GetBaseSymbol() const {
    return baseExpr ? baseExpr->GetBaseSymbol() : NULL;
}


/** Utility routine that takes a regualr pointer (either uniform or
    varying) and returns a slice pointer with zero offsets.
 */
static llvm::Value *
lConvertToSlicePointer(FunctionEmitContext *ctx, llvm::Value *ptr,
                       const PointerType *slicePtrType) {
    llvm::Type *llvmSlicePtrType = 
        slicePtrType->LLVMType(g->ctx);
    llvm::StructType *sliceStructType =
        llvm::dyn_cast<llvm::StructType>(llvmSlicePtrType);
    Assert(sliceStructType != NULL &&
           sliceStructType->getElementType(0) == ptr->getType());

    // Get a null-initialized struct to take care of having zeros for the
    // offsets
    llvm::Value *result = llvm::Constant::getNullValue(sliceStructType);
    // And replace the pointer in the struct with the given pointer
    return ctx->InsertInst(result, ptr, 0, LLVMGetName(ptr, "_slice"));
}


/** If the given array index is a compile time constant, check to see if it
    value/values don't go past the end of the array; issue a warning if
    so.
*/
static void
lCheckIndicesVersusBounds(const Type *baseExprType, Expr *index) {
    const SequentialType *seqType = CastType<SequentialType>(baseExprType);
    if (seqType == NULL)
        return;

    int nElements = seqType->GetElementCount();
    if (nElements == 0)
        // Unsized array...
        return;

    // If it's an array of soa<> items, then the number of elements to
    // worry about w.r.t. index values is the product of the array size and
    // the soa width.
    int soaWidth = seqType->GetElementType()->GetSOAWidth();
    if (soaWidth > 0)
        nElements *= soaWidth;

    ConstExpr *ce = dynamic_cast<ConstExpr *>(index);
    if (ce == NULL)
        return;

    int32_t indices[ISPC_MAX_NVEC];
    int count = ce->AsInt32(indices);
    for (int i = 0; i < count; ++i) {
        if (indices[i] < 0 || indices[i] >= nElements)
            Warning(index->pos, "Array index \"%d\" may be out of bounds for %d "
                    "element array.", indices[i], nElements);
    }
}


/** Converts the given pointer value to a slice pointer if the pointer
    points to SOA'ed data. 
*/
static llvm::Value *
lConvertPtrToSliceIfNeeded(FunctionEmitContext *ctx, 
                           llvm::Value *ptr, const Type **type) {
    Assert(*type != NULL);
    const PointerType *ptrType = CastType<PointerType>(*type);
    bool convertToSlice = (ptrType->GetBaseType()->IsSOAType() &&
                           ptrType->IsSlice() == false);
    if (convertToSlice == false)
        return ptr;

    *type = ptrType->GetAsSlice();
    return lConvertToSlicePointer(ctx, ptr, ptrType->GetAsSlice());
}


llvm::Value *
IndexExpr::GetLValue(FunctionEmitContext *ctx) const {
    const Type *baseExprType;
    if (baseExpr == NULL || index == NULL || 
        ((baseExprType = baseExpr->GetType()) == NULL)) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    ctx->SetDebugPos(pos);
    llvm::Value *indexValue = index->GetValue(ctx);
    if (indexValue == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    ctx->SetDebugPos(pos);
    if (CastType<PointerType>(baseExprType) != NULL) {
        // We're indexing off of a pointer 
        llvm::Value *basePtrValue = baseExpr->GetValue(ctx);
        if (basePtrValue == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }
        ctx->SetDebugPos(pos);

        // Convert to a slice pointer if we're indexing into SOA data
        basePtrValue = lConvertPtrToSliceIfNeeded(ctx, basePtrValue,
                                                  &baseExprType);

        llvm::Value *ptr = ctx->GetElementPtrInst(basePtrValue, indexValue,
                                                  baseExprType, 
                                                  LLVMGetName(basePtrValue, "_offset"));
        return lAddVaryingOffsetsIfNeeded(ctx, ptr, GetLValueType());
    }

    // Not a pointer: we must be indexing an array or vector (and possibly
    // a reference thereuponfore.)
    llvm::Value *basePtr = NULL;
    const PointerType *basePtrType = NULL;
    if (CastType<ArrayType>(baseExprType) ||
        CastType<VectorType>(baseExprType)) {
        basePtr = baseExpr->GetLValue(ctx);
        basePtrType = CastType<PointerType>(baseExpr->GetLValueType());
        if (baseExpr->GetLValueType()) AssertPos(pos, basePtrType != NULL);
    }
    else {
        baseExprType = baseExprType->GetReferenceTarget();
        AssertPos(pos, CastType<ArrayType>(baseExprType) ||
               CastType<VectorType>(baseExprType));
        basePtr = baseExpr->GetValue(ctx);
        basePtrType = PointerType::GetUniform(baseExprType);
    }
    if (!basePtr)
        return NULL;

    // If possible, check the index value(s) against the size of the array
    lCheckIndicesVersusBounds(baseExprType, index);

    // Convert to a slice pointer if indexing into SOA data
    basePtr = lConvertPtrToSliceIfNeeded(ctx, basePtr, 
                                         (const Type **)&basePtrType);

    ctx->SetDebugPos(pos);

    // And do the actual indexing calculation..
    llvm::Value *ptr = 
        ctx->GetElementPtrInst(basePtr, LLVMInt32(0), indexValue, 
                               basePtrType, LLVMGetName(basePtr, "_offset"));
    return lAddVaryingOffsetsIfNeeded(ctx, ptr, GetLValueType());
}


const Type *
IndexExpr::GetLValueType() const {
    if (lvalueType != NULL)
        return lvalueType;

    const Type *baseExprType, *baseExprLValueType, *indexType;
    if (baseExpr == NULL || index == NULL ||
        ((baseExprType = baseExpr->GetType()) == NULL) ||
        ((baseExprLValueType = baseExpr->GetLValueType()) == NULL) ||
        ((indexType = index->GetType()) == NULL))
        return NULL;

    // regularize to a PointerType 
    if (CastType<ReferenceType>(baseExprLValueType) != NULL) {
        const Type *refTarget = baseExprLValueType->GetReferenceTarget();
        baseExprLValueType = PointerType::GetUniform(refTarget);
    }
    AssertPos(pos, CastType<PointerType>(baseExprLValueType) != NULL);

    // Find the type of thing that we're indexing into
    const Type *elementType;
    const SequentialType *st = 
        CastType<SequentialType>(baseExprLValueType->GetBaseType());
    if (st != NULL)
        elementType = st->GetElementType();
    else {
        const PointerType *pt = 
            CastType<PointerType>(baseExprLValueType->GetBaseType());
        AssertPos(pos, pt != NULL);
        elementType = pt->GetBaseType();
    }

    // Are we indexing into a varying type, or are we indexing with a
    // varying pointer?
    bool baseVarying;
    if (CastType<PointerType>(baseExprType) != NULL)
        baseVarying = baseExprType->IsVaryingType();
    else
        baseVarying = baseExprLValueType->IsVaryingType();

    // The return type is uniform iff. the base is a uniform pointer / a
    // collection of uniform typed elements and the index is uniform.
    if (baseVarying == false && indexType->IsUniformType())
        lvalueType = PointerType::GetUniform(elementType);
    else
        lvalueType = PointerType::GetVarying(elementType);

    // Finally, if we're indexing into an SOA type, then the resulting
    // pointer must (currently) be a slice pointer; we don't allow indexing
    // the soa-width-wide structs directly.
    if (elementType->IsSOAType())
        lvalueType = lvalueType->GetAsSlice();

    return lvalueType;
}


Expr *
IndexExpr::Optimize() {
    if (baseExpr == NULL || index == NULL)
        return NULL;
    return this;
}


Expr *
IndexExpr::TypeCheck() {
    const Type *indexType;
    if (baseExpr == NULL || index == NULL || 
        ((indexType = index->GetType()) == NULL)) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    const Type *baseExprType = baseExpr->GetType();
    if (baseExprType == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    if (!CastType<SequentialType>(baseExprType->GetReferenceTarget())) {
        if (const PointerType *pt = CastType<PointerType>(baseExprType)) {
            if (Type::Equal(AtomicType::Void, pt->GetBaseType())) {
                Error(pos, "Illegal to dereference void pointer type \"%s\".",
                      baseExprType->GetString().c_str());
                return NULL;
            }
        }
        else {
            Error(pos, "Trying to index into non-array, vector, or pointer "
              "type \"%s\".", baseExprType->GetString().c_str());
            return NULL;
        }
    }

    bool isUniform = (index->GetType()->IsUniformType() && 
                      !g->opt.disableUniformMemoryOptimizations);

    if (!isUniform) {
        // Unless we have an explicit 64-bit index and are compiling to a
        // 64-bit target with 64-bit addressing, convert the index to an int32
        // type.
        //    The range of varying index is limited to [0,2^31) as a result.
        if (Type::EqualIgnoringConst(indexType->GetAsUniformType(),
                                     AtomicType::UniformInt64) == false ||
            g->target.is32Bit ||
            g->opt.force32BitAddressing) {
            const Type *indexType = AtomicType::VaryingInt32;
            index = TypeConvertExpr(index, indexType, "array index");
            if (index == NULL)
                return NULL;
        }
    } else { // isUniform
        // For 32-bit target:
        //   force the index to 32 bit.
        // For 64-bit target:
        //   We don't want to limit the index range. 
        //   We sxt/zxt the index to 64 bit right here because 
        //   LLVM doesn't distinguish unsigned from signed (both are i32)
        //
        //   However, the index can be still truncated to signed int32 if
        //   the index type is 64 bit and --addressing=32.
        bool force_32bit = g->target.is32Bit || 
            (g->opt.force32BitAddressing && 
             Type::EqualIgnoringConst(indexType->GetAsUniformType(),
                                      AtomicType::UniformInt64));
        const Type *indexType = force_32bit ? 
            AtomicType::UniformInt32 : AtomicType::UniformInt64;
        index = TypeConvertExpr(index, indexType, "array index");
        if (index == NULL)
            return NULL;
    }

    return this;
}


int
IndexExpr::EstimateCost() const {
    if (index == NULL || baseExpr == NULL)
        return 0;

    const Type *indexType = index->GetType();
    const Type *baseExprType = baseExpr->GetType();
    
    if ((indexType != NULL && indexType->IsVaryingType()) ||
        (CastType<PointerType>(baseExprType) != NULL &&
         baseExprType->IsVaryingType()))
        // be pessimistic; some of these will later turn out to be vector
        // loads/stores, but it's too early for us to know that here.
        return COST_GATHER;
    else
        return COST_LOAD;
}


void
IndexExpr::Print() const {
    if (!baseExpr || !index || !GetType())
        return;

    printf("[%s] index ", GetType()->GetString().c_str());
    baseExpr->Print();
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

//////////////////////////////////////////////////
// StructMemberExpr

class StructMemberExpr : public MemberExpr
{
public:
    StructMemberExpr(Expr *e, const char *id, SourcePos p,
                     SourcePos idpos, bool derefLValue);

    const Type *GetType() const;
    const Type *GetLValueType() const;
    int getElementNumber() const;
    const Type *getElementType() const;

private:
    const StructType *getStructType() const;
};


StructMemberExpr::StructMemberExpr(Expr *e, const char *id, SourcePos p,
                                   SourcePos idpos, bool derefLValue)
    : MemberExpr(e, id, p, idpos, derefLValue) {
}


const Type *
StructMemberExpr::GetType() const {
    if (type != NULL)
        return type;

    // It's a struct, and the result type is the element type, possibly
    // promoted to varying if the struct type / lvalue is varying.
    const Type *exprType, *lvalueType;
    const StructType *structType;
    if (expr == NULL ||
        ((exprType = expr->GetType()) == NULL) ||
        ((structType = getStructType()) == NULL) ||
        ((lvalueType = GetLValueType()) == NULL)) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    const Type *elementType = structType->GetElementType(identifier);
    if (elementType == NULL) {
        Error(identifierPos,
              "Element name \"%s\" not present in struct type \"%s\".%s",
              identifier.c_str(), structType->GetString().c_str(),
              getCandidateNearMatches().c_str());
        return NULL;
    }
    AssertPos(pos, Type::Equal(lvalueType->GetBaseType(), elementType));

    bool isSlice = (CastType<PointerType>(lvalueType) &&
                    CastType<PointerType>(lvalueType)->IsSlice());
    if (isSlice) {
        // FIXME: not true if we allow bound unif/varying for soa<>
        // structs?...
        AssertPos(pos, elementType->IsSOAType());

        // If we're accessing a member of an soa structure via a uniform
        // slice pointer, then the result type is the uniform variant of
        // the element type.
        if (lvalueType->IsUniformType())
            elementType = elementType->GetAsUniformType();
    }

    if (lvalueType->IsVaryingType())
        // If the expression we're getting the member of has an lvalue that
        // is a varying pointer type (be it slice or non-slice), then the
        // result type must be the varying version of the element type.
        elementType = elementType->GetAsVaryingType();

    type = elementType;
    return type;
}


const Type *
StructMemberExpr::GetLValueType() const {
    if (lvalueType != NULL)
        return lvalueType;

    if (expr == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    const Type *exprLValueType = dereferenceExpr ? expr->GetType() :
        expr->GetLValueType();
    if (exprLValueType == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    // The pointer type is varying if the lvalue type of the expression is
    // varying (and otherwise uniform)
    const PointerType *ptrType =
        (exprLValueType->IsUniformType() ||
         CastType<ReferenceType>(exprLValueType) != NULL) ?
        PointerType::GetUniform(getElementType()) : 
        PointerType::GetVarying(getElementType());

    // If struct pointer is a slice pointer, the resulting member pointer
    // needs to be a frozen slice pointer--i.e. any further indexing with
    // the result shouldn't modify the minor slice offset, but it should be
    // left unchanged until we get to a leaf SOA value.
    if (CastType<PointerType>(exprLValueType) &&
        CastType<PointerType>(exprLValueType)->IsSlice())
        ptrType = ptrType->GetAsFrozenSlice();

    lvalueType = ptrType;
    return lvalueType;
}


int
StructMemberExpr::getElementNumber() const {
    const StructType *structType = getStructType();
    if (structType == NULL)
        return -1;

    int elementNumber = structType->GetElementNumber(identifier);
    if (elementNumber == -1)
        Error(identifierPos,
              "Element name \"%s\" not present in struct type \"%s\".%s",
              identifier.c_str(), structType->GetString().c_str(),
              getCandidateNearMatches().c_str());

    return elementNumber;
}


const Type *
StructMemberExpr::getElementType() const {
    const StructType *structType = getStructType();
    if (structType == NULL)
        return NULL;

    return structType->GetElementType(identifier);
}


/** Returns the type of the underlying struct that we're returning a member
    of. */
const StructType *
StructMemberExpr::getStructType() const {
    const Type *type = dereferenceExpr ? expr->GetType() :
        expr->GetLValueType();
    if (type == NULL)
        return NULL;

    const Type *structType;
    const ReferenceType *rt = CastType<ReferenceType>(type);
    if (rt != NULL)
        structType = rt->GetReferenceTarget();
    else {
        const PointerType *pt = CastType<PointerType>(type);
        AssertPos(pos, pt != NULL);
        structType = pt->GetBaseType();
    }

    const StructType *ret = CastType<StructType>(structType);
    AssertPos(pos, ret != NULL);
    return ret;
}


//////////////////////////////////////////////////
// VectorMemberExpr

class VectorMemberExpr : public MemberExpr
{
public:
    VectorMemberExpr(Expr *e, const char *id, SourcePos p,
                     SourcePos idpos, bool derefLValue);

    llvm::Value *GetValue(FunctionEmitContext* ctx) const;
    llvm::Value *GetLValue(FunctionEmitContext* ctx) const;
    const Type *GetType() const;
    const Type *GetLValueType() const;

    int getElementNumber() const;
    const Type *getElementType() const;

private:
    const VectorType *exprVectorType;
    const VectorType *memberType;
};


VectorMemberExpr::VectorMemberExpr(Expr *e, const char *id, SourcePos p,
                                   SourcePos idpos, bool derefLValue)
    : MemberExpr(e, id, p, idpos, derefLValue) {
    const Type *exprType = e->GetType();
    exprVectorType = CastType<VectorType>(exprType);
    if (exprVectorType == NULL) {
        const PointerType *pt = CastType<PointerType>(exprType);
        if (pt != NULL)
            exprVectorType = CastType<VectorType>(pt->GetBaseType());
        else {
            AssertPos(pos, CastType<ReferenceType>(exprType) != NULL);
            exprVectorType = 
                CastType<VectorType>(exprType->GetReferenceTarget());
        }
        AssertPos(pos, exprVectorType != NULL);
    }
    memberType = new VectorType(exprVectorType->GetElementType(),
                                identifier.length());
}


const Type *
VectorMemberExpr::GetType() const {
    if (type != NULL)
        return type;

    // For 1-element expressions, we have the base vector element
    // type.  For n-element expressions, we have a shortvec type
    // with n > 1 elements.  This can be changed when we get
    // type<1> -> type conversions.
    type = (identifier.length() == 1) ? 
        (const Type *)exprVectorType->GetElementType() : 
        (const Type *)memberType;

    const Type *lvType = GetLValueType();
    if (lvType != NULL) {
        bool isSlice = (CastType<PointerType>(lvType) &&
                        CastType<PointerType>(lvType)->IsSlice());
        if (isSlice) {
//CO            AssertPos(pos, type->IsSOAType());
            if (lvType->IsUniformType())
                type = type->GetAsUniformType();
        }

        if (lvType->IsVaryingType())
            type = type->GetAsVaryingType();
    }

    return type;
}


llvm::Value *
VectorMemberExpr::GetLValue(FunctionEmitContext* ctx) const {
    if (identifier.length() == 1) {
        return MemberExpr::GetLValue(ctx);
    } else {
        return NULL;
    }
}


const Type *
VectorMemberExpr::GetLValueType() const {
    if (lvalueType != NULL)
        return lvalueType;

    if (identifier.length() == 1) {
        if (expr == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        const Type *exprLValueType = dereferenceExpr ? expr->GetType() :
            expr->GetLValueType();
        if (exprLValueType == NULL)
            return NULL;

        const VectorType *vt = NULL;
        if (CastType<ReferenceType>(exprLValueType) != NULL)
            vt = CastType<VectorType>(exprLValueType->GetReferenceTarget());
        else
            vt = CastType<VectorType>(exprLValueType->GetBaseType());
        AssertPos(pos, vt != NULL);

        // we don't want to report that it's e.g. a pointer to a float<1>,
        // but a pointer to a float, etc.
        const Type *elementType = vt->GetElementType();
        if (CastType<ReferenceType>(exprLValueType) != NULL)
            lvalueType = new ReferenceType(elementType);
        else {
            const PointerType *ptrType = exprLValueType->IsUniformType() ?
                PointerType::GetUniform(elementType) : 
                PointerType::GetVarying(elementType);
            // FIXME: replicated logic with structmemberexpr.... 
            if (CastType<PointerType>(exprLValueType) &&
                CastType<PointerType>(exprLValueType)->IsSlice())
                ptrType = ptrType->GetAsFrozenSlice();
            lvalueType = ptrType;
        }
    }

    return lvalueType;
}


llvm::Value *
VectorMemberExpr::GetValue(FunctionEmitContext *ctx) const {
    if (identifier.length() == 1) {
        return MemberExpr::GetValue(ctx);
    } 
    else {
        std::vector<int> indices;

        for (size_t i = 0; i < identifier.size(); ++i) {
            int idx = lIdentifierToVectorElement(identifier[i]);
            if (idx == -1)
                Error(pos,
                      "Invalid swizzle charcter '%c' in swizzle \"%s\".",
                      identifier[i], identifier.c_str());

            indices.push_back(idx);
        }

        llvm::Value *basePtr = NULL;
        const Type *basePtrType = NULL;
        if (dereferenceExpr) {
            basePtr = expr->GetValue(ctx);
            basePtrType = expr->GetType();
        }
        else {
            basePtr = expr->GetLValue(ctx);
            basePtrType = expr->GetLValueType();
        }

        if (basePtr == NULL || basePtrType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        // Allocate temporary memory to tore the result
        llvm::Value *resultPtr = ctx->AllocaInst(memberType->LLVMType(g->ctx), 
                                                 "vector_tmp");
        
        // FIXME: we should be able to use the internal mask here according
        // to the same logic where it's used elsewhere
        llvm::Value *elementMask = ctx->GetFullMask();

        const Type *elementPtrType = basePtrType->IsUniformType() ? 
            PointerType::GetUniform(exprVectorType->GetElementType()) :
            PointerType::GetVarying(exprVectorType->GetElementType());

        ctx->SetDebugPos(pos);
        for (size_t i = 0; i < identifier.size(); ++i) {
            char idStr[2] = { identifier[i], '\0' };
            llvm::Value *elementPtr = ctx->AddElementOffset(basePtr, indices[i],
                                                            basePtrType,
                                                            LLVMGetName(basePtr, idStr));
            llvm::Value *elementValue = 
                ctx->LoadInst(elementPtr, elementMask, elementPtrType);

            const char *resultName = LLVMGetName(resultPtr, idStr);
            llvm::Value *ptmp = ctx->AddElementOffset(resultPtr, i, NULL, resultName);
            ctx->StoreInst(elementValue, ptmp);
        }

        return ctx->LoadInst(resultPtr, LLVMGetName(basePtr, "_swizzle"));
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


const Type *
VectorMemberExpr::getElementType() const {
    return memberType;
}



MemberExpr *
MemberExpr::create(Expr *e, const char *id, SourcePos p, SourcePos idpos,
                   bool derefLValue) {
    // FIXME: we need to call TypeCheck() here so that we can call
    // e->GetType() in the following.  But really we just shouldn't try to
    // resolve this now but just have a generic MemberExpr type that
    // handles all cases so that this is unnecessary.
    e = ::TypeCheck(e);

    const Type *exprType;
    if (e == NULL || (exprType = e->GetType()) == NULL)
        return NULL;

    const ReferenceType *referenceType = CastType<ReferenceType>(exprType);
    if (referenceType != NULL) {
        e = new RefDerefExpr(e, e->pos);
        exprType = e->GetType();
        Assert(exprType != NULL);
    }

    const PointerType *pointerType = CastType<PointerType>(exprType);
    if (pointerType != NULL)
        exprType = pointerType->GetBaseType();

    if (derefLValue == true && pointerType == NULL) {
        const Type *targetType = exprType->GetReferenceTarget();
        if (CastType<StructType>(targetType) != NULL)
            Error(p, "Member operator \"->\" can't be applied to non-pointer "
                  "type \"%s\".  Did you mean to use \".\"?", 
                  exprType->GetString().c_str());
        else
            Error(p, "Member operator \"->\" can't be applied to non-struct "
                  "pointer type \"%s\".", exprType->GetString().c_str());
        return NULL;
    }
    if (derefLValue == false && pointerType != NULL &&
        CastType<StructType>(pointerType->GetBaseType()) != NULL) {
            Error(p, "Member operator \".\" can't be applied to pointer "
                  "type \"%s\".  Did you mean to use \"->\"?", 
                  exprType->GetString().c_str());
        return NULL;
    }

    if (CastType<StructType>(exprType) != NULL)
        return new StructMemberExpr(e, id, p, idpos, derefLValue);
    else if (CastType<VectorType>(exprType) != NULL)
        return new VectorMemberExpr(e, id, p, idpos, derefLValue);
    else if (CastType<UndefinedStructType>(exprType)) {
        Error(p, "Member operator \"%s\" can't be applied to declared "
              "but not defined struct type \"%s\".", derefLValue ? "->" : ".",
              exprType->GetString().c_str());
        return NULL;
    }
    else {
        Error(p, "Member operator \"%s\" can't be used with expression of "
              "\"%s\" type.", derefLValue ? "->" : ".", 
              exprType->GetString().c_str());
        return NULL;
    }
}


MemberExpr::MemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos,
                       bool derefLValue) 
    : Expr(p), identifierPos(idpos) {
    expr = e;
    identifier = id;
    dereferenceExpr = derefLValue;
    type = lvalueType = NULL;
}


llvm::Value *
MemberExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr) 
        return NULL;

    llvm::Value *lvalue = GetLValue(ctx);
    const Type *lvalueType = GetLValueType();

    llvm::Value *mask = NULL;
    if (lvalue == NULL) {
        if (m->errorCount > 0)
            return NULL;

        // As in the array case, this may be a temporary that hasn't hit
        // memory; get the full value and stuff it into a temporary array
        // so that we can index from there...
        llvm::Value *val = expr->GetValue(ctx);
        if (!val) {
            AssertPos(pos, m->errorCount > 0);
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

        lvalue = ctx->AddElementOffset(ptr, elementNumber, 
                                       PointerType::GetUniform(exprType));
        lvalueType = PointerType::GetUniform(GetType());
        mask = LLVMMaskAllOn;
    }
    else {
        Symbol *baseSym = GetBaseSymbol();
        AssertPos(pos, baseSym != NULL);
        mask = lMaskForSymbol(baseSym, ctx);
    }

    ctx->SetDebugPos(pos);
    std::string suffix = std::string("_") + identifier;
    return ctx->LoadInst(lvalue, mask, lvalueType, 
                         LLVMGetName(lvalue, suffix.c_str()));
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
    const Type *exprType;
    if (!expr || ((exprType = expr->GetType()) == NULL))
        return NULL;

    ctx->SetDebugPos(pos);
    llvm::Value *basePtr = dereferenceExpr ? expr->GetValue(ctx) :
        expr->GetLValue(ctx);
    if (!basePtr)
        return NULL;

    int elementNumber = getElementNumber();
    if (elementNumber == -1)
        return NULL;

    const Type *exprLValueType = dereferenceExpr ? expr->GetType() :
        expr->GetLValueType();
    ctx->SetDebugPos(pos);
    llvm::Value *ptr = ctx->AddElementOffset(basePtr, elementNumber,
                                             exprLValueType, 
                                             basePtr->getName().str().c_str());
    if (ptr == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    ptr = lAddVaryingOffsetsIfNeeded(ctx, ptr, GetLValueType());

    return ptr;
}


Expr *
MemberExpr::TypeCheck() {
    return expr ? this : NULL;
}


Expr *
MemberExpr::Optimize() {
    return expr ? this : NULL;
}


int
MemberExpr::EstimateCost() const {
    const Type *lvalueType = GetLValueType();
    if (lvalueType != NULL && lvalueType->IsVaryingType())
        return COST_GATHER + COST_SIMPLE_ARITH_LOGIC_OP;
    else
        return COST_SIMPLE_ARITH_LOGIC_OP;
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
    const StructType *structType = CastType<StructType>(expr->GetType());
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
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt8->GetAsConstType()));
    int8Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int8_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt8->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingInt8->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int8Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint8_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt8->GetAsConstType()));
    uint8Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint8_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt8->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingUInt8->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        uint8Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, int16_t i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt16->GetAsConstType()));
    int16Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int16_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt16->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingInt16->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int16Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint16_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt16->GetAsConstType()));
    uint16Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint16_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt16->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingUInt16->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        uint16Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, int32_t i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt32->GetAsConstType()));
    int32Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int32_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt32->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingInt32->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int32Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint32_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt32->GetAsConstType()) ||
           (CastType<EnumType>(type) != NULL &&
            type->IsUniformType()));
    uint32Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint32_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt32->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingUInt32->GetAsConstType()) ||
           (CastType<EnumType>(type) != NULL));
    for (int j = 0; j < Count(); ++j)
        uint32Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, float f, SourcePos p)
    : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformFloat->GetAsConstType()));
    floatVal[0] = f;
}


ConstExpr::ConstExpr(const Type *t, float *f, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformFloat->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingFloat->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        floatVal[j] = f[j];
}


ConstExpr::ConstExpr(const Type *t, int64_t i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt64->GetAsConstType()));
    int64Val[0] = i;
}


ConstExpr::ConstExpr(const Type *t, int64_t *i, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt64->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingInt64->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int64Val[j] = i[j];
}


ConstExpr::ConstExpr(const Type *t, uint64_t u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt64->GetAsConstType()));
    uint64Val[0] = u;
}


ConstExpr::ConstExpr(const Type *t, uint64_t *u, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt64->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingUInt64->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        uint64Val[j] = u[j];
}


ConstExpr::ConstExpr(const Type *t, double f, SourcePos p)
    : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformDouble->GetAsConstType()));
    doubleVal[0] = f;
}


ConstExpr::ConstExpr(const Type *t, double *f, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformDouble->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingDouble->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        doubleVal[j] = f[j];
}


ConstExpr::ConstExpr(const Type *t, bool b, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformBool->GetAsConstType()));
    boolVal[0] = b;
}


ConstExpr::ConstExpr(const Type *t, bool *b, SourcePos p) 
  : Expr(p) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformBool->GetAsConstType()) || 
           Type::Equal(type, AtomicType::VaryingBool->GetAsConstType()));
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


ConstExpr::ConstExpr(ConstExpr *old, SourcePos p)
    : Expr(p) {
    type = old->type;

    AtomicType::BasicType basicType = getBasicType();

    switch (basicType) {
    case AtomicType::TYPE_BOOL:
        memcpy(boolVal, old->boolVal, Count() * sizeof(bool));
        break;
    case AtomicType::TYPE_INT8:
        memcpy(int8Val, old->int8Val, Count() * sizeof(int8_t));
        break;
    case AtomicType::TYPE_UINT8:
        memcpy(uint8Val, old->uint8Val, Count() * sizeof(uint8_t));
        break;
    case AtomicType::TYPE_INT16:
        memcpy(int16Val, old->int16Val, Count() * sizeof(int16_t));
        break;
    case AtomicType::TYPE_UINT16:
        memcpy(uint16Val, old->uint16Val, Count() * sizeof(uint16_t));
        break;
    case AtomicType::TYPE_INT32:
        memcpy(int32Val, old->int32Val, Count() * sizeof(int32_t));
        break;
    case AtomicType::TYPE_UINT32:
        memcpy(uint32Val, old->uint32Val, Count() * sizeof(uint32_t));
        break;
    case AtomicType::TYPE_FLOAT:
        memcpy(floatVal, old->floatVal, Count() * sizeof(float));
        break;
    case AtomicType::TYPE_DOUBLE:
        memcpy(doubleVal, old->doubleVal, Count() * sizeof(double));
        break;
    case AtomicType::TYPE_INT64:
        memcpy(int64Val, old->int64Val, Count() * sizeof(int64_t));
        break;
    case AtomicType::TYPE_UINT64:
        memcpy(uint64Val, old->uint64Val, Count() * sizeof(uint64_t));
        break;
    default:
        FATAL("unimplemented const type");
    }
    
}


AtomicType::BasicType
ConstExpr::getBasicType() const {
    const AtomicType *at = CastType<AtomicType>(type);
    if (at != NULL)
        return at->basicType;
    else {
        AssertPos(pos, CastType<EnumType>(type) != NULL);
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
        AssertPos(pos, Count() == 1);

    type = type->GetAsNonConstType();
    if (Type::Equal(type, AtomicType::UniformBool) || 
        Type::Equal(type, AtomicType::VaryingBool)) {
        bool bv[ISPC_MAX_NVEC];
        AsBool(bv, type->IsVaryingType());
        if (type->IsUniformType())
            return bv[0] ? LLVMTrue : LLVMFalse;
        else
            return LLVMBoolVector(bv);
    }
    else if (Type::Equal(type, AtomicType::UniformInt8) ||
             Type::Equal(type, AtomicType::VaryingInt8)) {
        int8_t iv[ISPC_MAX_NVEC];
        AsInt8(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt8(iv[0]);
        else
            return LLVMInt8Vector(iv);
    }
    else if (Type::Equal(type, AtomicType::UniformUInt8) || 
             Type::Equal(type, AtomicType::VaryingUInt8)) {
        uint8_t uiv[ISPC_MAX_NVEC];
        AsUInt8(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt8(uiv[0]);
        else
            return LLVMUInt8Vector(uiv);
    }
    else if (Type::Equal(type, AtomicType::UniformInt16) ||
             Type::Equal(type, AtomicType::VaryingInt16)) {
        int16_t iv[ISPC_MAX_NVEC];
        AsInt16(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt16(iv[0]);
        else
            return LLVMInt16Vector(iv);
    }
    else if (Type::Equal(type, AtomicType::UniformUInt16) ||
             Type::Equal(type, AtomicType::VaryingUInt16)) {
        uint16_t uiv[ISPC_MAX_NVEC];
        AsUInt16(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt16(uiv[0]);
        else
            return LLVMUInt16Vector(uiv);
    }
    else if (Type::Equal(type, AtomicType::UniformInt32) || 
             Type::Equal(type, AtomicType::VaryingInt32)) {
        int32_t iv[ISPC_MAX_NVEC];
        AsInt32(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt32(iv[0]);
        else
            return LLVMInt32Vector(iv);
    }
    else if (Type::Equal(type, AtomicType::UniformUInt32) || 
             Type::Equal(type, AtomicType::VaryingUInt32) ||
             CastType<EnumType>(type) != NULL) {
        uint32_t uiv[ISPC_MAX_NVEC];
        AsUInt32(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt32(uiv[0]);
        else
            return LLVMUInt32Vector(uiv);
    }
    else if (Type::Equal(type, AtomicType::UniformFloat) || 
             Type::Equal(type, AtomicType::VaryingFloat)) {
        float fv[ISPC_MAX_NVEC];
        AsFloat(fv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMFloat(fv[0]);
        else
            return LLVMFloatVector(fv);
    }
    else if (Type::Equal(type, AtomicType::UniformInt64) || 
             Type::Equal(type, AtomicType::VaryingInt64)) {
        int64_t iv[ISPC_MAX_NVEC];
        AsInt64(iv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMInt64(iv[0]);
        else
            return LLVMInt64Vector(iv);
    }
    else if (Type::Equal(type, AtomicType::UniformUInt64) ||
             Type::Equal(type, AtomicType::VaryingUInt64)) {
        uint64_t uiv[ISPC_MAX_NVEC];
        AsUInt64(uiv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMUInt64(uiv[0]);
        else
            return LLVMUInt64Vector(uiv);
    }
    else if (Type::Equal(type, AtomicType::UniformDouble) ||
             Type::Equal(type, AtomicType::VaryingDouble)) {
        double dv[ISPC_MAX_NVEC];
        AsDouble(dv, type->IsVaryingType());
        if (type->IsUniformType())
            return LLVMDouble(dv[0]);
        else
            return LLVMDoubleVector(dv);
    }
    else if (CastType<PointerType>(type) != NULL) {
        // The only time we should get here is if we have an integer '0'
        // constant that should be turned into a NULL pointer of the
        // appropriate type.
        llvm::Type *llvmType = type->LLVMType(g->ctx);
        if (llvmType == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }

        int64_t iv[ISPC_MAX_NVEC];
        AsInt64(iv, type->IsVaryingType());
        for (int i = 0; i < Count(); ++i)
            if (iv[i] != 0)
                // We'll issue an error about this later--trying to assign
                // a constant int to a pointer, without a typecast.
                return NULL;

        return llvm::Constant::getNullValue(llvmType);
    }
    else {
        Debug(pos, "Unable to handle type \"%s\" in ConstExpr::GetConstant().",
              type->GetString().c_str());
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


int
ConstExpr::EstimateCost() const {
    return 0;
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
            printf("%"PRId64, int64Val[i]);
            break;
        case AtomicType::TYPE_UINT64:
            printf("%"PRIu64, uint64Val[i]);
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

    std::string opName = exprVal->getName().str();
    switch (toType->basicType) {
    case AtomicType::TYPE_BOOL: opName += "_to_bool"; break;
    case AtomicType::TYPE_INT8: opName += "_to_int8"; break;
    case AtomicType::TYPE_UINT8: opName += "_to_uint8"; break;
    case AtomicType::TYPE_INT16: opName += "_to_int16"; break;
    case AtomicType::TYPE_UINT16: opName += "_to_uint16"; break;
    case AtomicType::TYPE_INT32: opName += "_to_int32"; break;
    case AtomicType::TYPE_UINT32: opName += "_to_uint32"; break;
    case AtomicType::TYPE_INT64: opName += "_to_int64"; break;
    case AtomicType::TYPE_UINT64: opName += "_to_uint64"; break;
    case AtomicType::TYPE_FLOAT: opName += "_to_float"; break;
    case AtomicType::TYPE_DOUBLE: opName += "_to_double"; break;
    default: FATAL("Unimplemented");
    }
    const char *cOpName = opName.c_str();

    switch (toType->basicType) {
    case AtomicType::TYPE_FLOAT: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::FloatType : 
                                        LLVMTypes::FloatVectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                // If we have a bool vector of i32 elements, first truncate
                // down to a single bit
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            // And then do an unisgned int->float cast
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_INT64:
            cast = ctx->CastInst(llvm::Instruction::SIToFP, // signed int to float
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_UINT64:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from unsigned int to float is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            // No-op cast.
            cast = exprVal;
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->FPCastInst(exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_DOUBLE: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::DoubleType :
                                        LLVMTypes::DoubleVectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                // truncate i32 bool vector values to i1s
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to double
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_INT64:
            cast = ctx->CastInst(llvm::Instruction::SIToFP, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_UINT64:
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->FPCastInst(exprVal, targetType, cOpName);
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
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int8Type :
                                        LLVMTypes::Int8VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
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
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT8: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int8Type :
                                        LLVMTypes::Int8VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
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
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from float to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from double to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT16: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int16Type :
                                        LLVMTypes::Int16VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
            cast = ctx->SExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT16: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int16Type :
                                        LLVMTypes::Int16VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
            cast = ctx->SExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;            
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from float to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from double to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT32: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int32Type :
                                        LLVMTypes::Int32VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
            cast = ctx->SExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT32: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int32Type :
                                        LLVMTypes::Int32VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
            cast = ctx->SExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;            
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from float to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from double to unsigned int is slow. "
                                   "Use \"int\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT64: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int64Type : 
                                        LLVMTypes::Int64VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() &&
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
            cast = ctx->SExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = exprVal;
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToSI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_UINT64: {
        llvm::Type *targetType = 
            fromType->IsUniformType() ? LLVMTypes::Int64Type : 
                                        LLVMTypes::Int64VectorType;
        switch (fromType->basicType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingType() && 
                LLVMTypes::BoolVectorType == LLVMTypes::Int32VectorType)
                exprVal = ctx->TruncInst(exprVal, LLVMTypes::Int1VectorType, cOpName);
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
            cast = ctx->SExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
        case AtomicType::TYPE_UINT32:
            cast = ctx->ZExtInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from float to unsigned int64 is slow. "
                                   "Use \"int64\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = exprVal;
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingType() && g->target.isa != Target::GENERIC)
                PerformanceWarning(pos, "Conversion from double to unsigned int64 is slow. "
                                   "Use \"int64\" if possible");
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // signed int
                                 exprVal, targetType, cOpName);
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
                                exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMInt16(0) : 
                (llvm::Value *)LLVMInt16Vector((int16_t)0);
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMInt32(0) : 
                (llvm::Value *)LLVMInt32Vector(0);
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_FLOAT: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMFloat(0.f) : 
                (llvm::Value *)LLVMFloatVector(0.f);
            cast = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_ONE,
                                exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMInt64(0) : 
                (llvm::Value *)LLVMInt64Vector((int64_t)0);
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_DOUBLE: {
            llvm::Value *zero = fromType->IsUniformType() ? (llvm::Value *)LLVMDouble(0.) : 
                (llvm::Value *)LLVMDoubleVector(0.);
            cast = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_ONE,
                                exprVal, zero, cOpName);
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
                                     LLVMGetName(cast, "to_i32bool"));
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
    if (toType->IsVaryingType() && fromType->IsUniformType())
        return ctx->SmearUniform(cast);
    else
        return cast;
}


// FIXME: fold this into the FunctionEmitContext::SmearUniform() method?

/** Converts the given value of the given type to be the varying
    equivalent, returning the resulting value.
 */
static llvm::Value *
lUniformValueToVarying(FunctionEmitContext *ctx, llvm::Value *value,
                       const Type *type) {
    // nothing to do if it's already varying
    if (type->IsVaryingType())
        return value;

    // for structs/arrays/vectors, just recursively make their elements
    // varying (if needed) and populate the return value.
    const CollectionType *collectionType = CastType<CollectionType>(type);
    if (collectionType != NULL) {
        llvm::Type *llvmType = 
            type->GetAsVaryingType()->LLVMType(g->ctx);
        llvm::Value *retValue = llvm::UndefValue::get(llvmType);

        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            llvm::Value *v = ctx->ExtractInst(value, i, "get_element");
            v = lUniformValueToVarying(ctx, v, collectionType->GetElementType(i));
            retValue = ctx->InsertInst(retValue, v, i, "set_element");
        }
        return retValue;
    }

    // Otherwise we must have a uniform atomic or pointer type, so smear
    // its value across the vector lanes.
    Assert(CastType<AtomicType>(type) != NULL ||
           CastType<PointerType>(type) != NULL);
    return ctx->SmearUniform(value);
}


llvm::Value *
TypeCastExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr)
        return NULL;

    ctx->SetDebugPos(pos);
    const Type *toType = GetType(), *fromType = expr->GetType();
    if (toType == NULL || fromType == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    if (Type::Equal(toType, AtomicType::Void)) {
        // emit the code for the expression in case it has side-effects but
        // then we're done.
        (void)expr->GetValue(ctx);
        return NULL;
    }

    const PointerType *fromPointerType = CastType<PointerType>(fromType);
    const PointerType *toPointerType = CastType<PointerType>(toType);
    const ArrayType *toArrayType = CastType<ArrayType>(toType);
    const ArrayType *fromArrayType = CastType<ArrayType>(fromType);
    if (fromPointerType != NULL) {
        if (toArrayType != NULL) {
            return expr->GetValue(ctx);
        }
        else if (toPointerType != NULL) {
            llvm::Value *value = expr->GetValue(ctx);
            if (value == NULL)
                return NULL;

            if (fromPointerType->IsSlice() == false &&
                toPointerType->IsSlice() == true) {
                // Convert from a non-slice pointer to a slice pointer by
                // creating a slice pointer structure with zero offsets.
                if (fromPointerType->IsUniformType())
                    value = ctx->MakeSlicePointer(value, LLVMInt32(0));
                else
                    value = ctx->MakeSlicePointer(value, LLVMInt32Vector(0));

                // FIXME: avoid error from unnecessary bitcast when all we
                // need to do is the slice conversion and don't need to
                // also do unif->varying conversions.  But this is really
                // ugly logic.
                if (value->getType() == toType->LLVMType(g->ctx))
                    return value;
            }

            if (fromType->IsUniformType() && toType->IsUniformType())
                // bitcast to the actual pointer type
                return ctx->BitCastInst(value, toType->LLVMType(g->ctx));
            else if (fromType->IsVaryingType() && toType->IsVaryingType()) {
                // both are vectors of ints already, nothing to do at the IR
                // level
                return value;
            }
            else {
                // Uniform -> varying pointer conversion
                AssertPos(pos, fromType->IsUniformType() && toType->IsVaryingType());
                if (fromPointerType->IsSlice()) {
                    // For slice pointers, we need to smear out both the
                    // pointer and the offset vector
                    AssertPos(pos, toPointerType->IsSlice());
                    llvm::Value *ptr = ctx->ExtractInst(value, 0);
                    llvm::Value *offset = ctx->ExtractInst(value, 1);
                    ptr = ctx->PtrToIntInst(ptr);
                    ptr = ctx->SmearUniform(ptr);
                    offset = ctx->SmearUniform(offset);
                    return ctx->MakeSlicePointer(ptr, offset);
                }
                else {
                    // Otherwise we just bitcast it to an int and smear it
                    // out to a vector
                    value = ctx->PtrToIntInst(value);
                    return ctx->SmearUniform(value);
                }
            }
        }
        else {
            AssertPos(pos, CastType<AtomicType>(toType) != NULL);
            if (toType->IsBoolType()) {
                // convert pointer to bool
                llvm::Type *lfu = 
                    fromType->GetAsUniformType()->LLVMType(g->ctx);
                llvm::PointerType *llvmFromUnifType = 
                    llvm::dyn_cast<llvm::PointerType>(lfu);

                llvm::Value *nullPtrValue = 
                    llvm::ConstantPointerNull::get(llvmFromUnifType);
                if (fromType->IsVaryingType())
                    nullPtrValue = ctx->SmearUniform(nullPtrValue);

                llvm::Value *exprVal = expr->GetValue(ctx);
                llvm::Value *cmp = 
                    ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE,
                                 exprVal, nullPtrValue, "ptr_ne_NULL");

                if (toType->IsVaryingType()) {
                    if (fromType->IsUniformType())
                        cmp = ctx->SmearUniform(cmp);
                    cmp = ctx->I1VecToBoolVec(cmp);
                }

                return cmp;
            }
            else {
                // ptr -> int
                llvm::Value *value = expr->GetValue(ctx);
                if (value == NULL)
                    return NULL;

                if (toType->IsVaryingType() && fromType->IsUniformType())
                    value = ctx->SmearUniform(value);

                llvm::Type *llvmToType = toType->LLVMType(g->ctx);
                if (llvmToType == NULL)
                    return NULL;
                return ctx->PtrToIntInst(value, llvmToType, "ptr_typecast");
            }
        }
    }

    if (Type::EqualIgnoringConst(toType, fromType))
        // There's nothing to do, just return the value.  (LLVM's type
        // system doesn't worry about constiness.)
        return expr->GetValue(ctx);

    if (fromArrayType != NULL && toPointerType != NULL) {
        // implicit array to pointer to first element
        Expr *arrayAsPtr = lArrayToPointer(expr);
        if (Type::EqualIgnoringConst(arrayAsPtr->GetType(), toPointerType) == false) {
            AssertPos(pos, PointerType::IsVoidPointer(toPointerType) ||
                   Type::EqualIgnoringConst(arrayAsPtr->GetType()->GetAsVaryingType(),
                                            toPointerType) == true);
            arrayAsPtr = new TypeCastExpr(toPointerType, arrayAsPtr, pos);
            arrayAsPtr = ::TypeCheck(arrayAsPtr);
            AssertPos(pos, arrayAsPtr != NULL);
            arrayAsPtr = ::Optimize(arrayAsPtr);
            AssertPos(pos, arrayAsPtr != NULL);
        }
        AssertPos(pos, Type::EqualIgnoringConst(arrayAsPtr->GetType(), toPointerType));
        return arrayAsPtr->GetValue(ctx);
    }

    // This also should be caught during typechecking
    AssertPos(pos, !(toType->IsUniformType() && fromType->IsVaryingType()));

    if (toArrayType != NULL && fromArrayType != NULL) {
        // cast array pointer from [n x foo] to [0 x foo] if needed to be able
        // to pass to a function that takes an unsized array as a parameter
        if (toArrayType->GetElementCount() != 0 && 
            (toArrayType->GetElementCount() != fromArrayType->GetElementCount()))
            Warning(pos, "Type-converting array of length %d to length %d",
                    fromArrayType->GetElementCount(), toArrayType->GetElementCount());
        AssertPos(pos, Type::EqualIgnoringConst(toArrayType->GetBaseType(),
                                        fromArrayType->GetBaseType()));
        llvm::Value *v = expr->GetValue(ctx);
        llvm::Type *ptype = toType->LLVMType(g->ctx);
        return ctx->BitCastInst(v, ptype); //, "array_cast_0size");
    }

    const ReferenceType *toReference = CastType<ReferenceType>(toType);
    const ReferenceType *fromReference = CastType<ReferenceType>(fromType);
    if (toReference && fromReference) {
        const Type *toTarget = toReference->GetReferenceTarget();
        const Type *fromTarget = fromReference->GetReferenceTarget();

        const ArrayType *toArray = CastType<ArrayType>(toTarget);
        const ArrayType *fromArray = CastType<ArrayType>(fromTarget);
        if (toArray && fromArray) {
            // cast array pointer from [n x foo] to [0 x foo] if needed to be able
            // to pass to a function that takes an unsized array as a parameter
            if(toArray->GetElementCount() != 0 && 
               (toArray->GetElementCount() != fromArray->GetElementCount()))
                Warning(pos, "Type-converting array of length %d to length %d",
                        fromArray->GetElementCount(), toArray->GetElementCount());
            AssertPos(pos, Type::EqualIgnoringConst(toArray->GetBaseType(),
                                            fromArray->GetBaseType()));
            llvm::Value *v = expr->GetValue(ctx);
            llvm::Type *ptype = toType->LLVMType(g->ctx);
            return ctx->BitCastInst(v, ptype); //, "array_cast_0size");
        }

        AssertPos(pos, Type::Equal(toTarget, fromTarget) ||
               Type::Equal(toTarget, fromTarget->GetAsConstType()));
        return expr->GetValue(ctx);
    }

    const StructType *toStruct = CastType<StructType>(toType);
    const StructType *fromStruct = CastType<StructType>(fromType);
    if (toStruct && fromStruct) {
        // The only legal type conversions for structs are to go from a
        // uniform to a varying instance of the same struct type.
        AssertPos(pos, toStruct->IsVaryingType() && fromStruct->IsUniformType() &&
               Type::EqualIgnoringConst(toStruct, 
                                        fromStruct->GetAsVaryingType()));

        llvm::Value *origValue = expr->GetValue(ctx);
        if (!origValue)
            return NULL;
        return lUniformValueToVarying(ctx, origValue, fromType);
    }

    const VectorType *toVector = CastType<VectorType>(toType);
    const VectorType *fromVector = CastType<VectorType>(fromType);
    if (toVector && fromVector) {
        // this should be caught during typechecking
        AssertPos(pos, toVector->GetElementCount() == fromVector->GetElementCount());

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

    const EnumType *fromEnum = CastType<EnumType>(fromType);
    const EnumType *toEnum = CastType<EnumType>(toType);
    if (fromEnum)
        // treat it as an uint32 type for the below and all will be good.
        fromType = fromEnum->IsUniformType() ? AtomicType::UniformUInt32 :
            AtomicType::VaryingUInt32;
    if (toEnum)
        // treat it as an uint32 type for the below and all will be good.
        toType = toEnum->IsUniformType() ? AtomicType::UniformUInt32 :
            AtomicType::VaryingUInt32;

    const AtomicType *fromAtomic = CastType<AtomicType>(fromType);
    // at this point, coming from an atomic type is all that's left...
    AssertPos(pos, fromAtomic != NULL);

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
    else if (toPointerType != NULL) {
        // int -> ptr
        if (toType->IsVaryingType() && fromType->IsUniformType())
            exprVal = ctx->SmearUniform(exprVal);

        llvm::Type *llvmToType = toType->LLVMType(g->ctx);
        if (llvmToType == NULL)
            return NULL;

        return ctx->IntToPtrInst(exprVal, llvmToType, "int_to_ptr");
    }
    else {
        const AtomicType *toAtomic = CastType<AtomicType>(toType);
        // typechecking should ensure this is the case
        AssertPos(pos, toAtomic != NULL);

        return lTypeConvAtomic(ctx, exprVal, toAtomic, fromAtomic, pos);
    }
}


const Type *
TypeCastExpr::GetType() const { 
    AssertPos(pos, type->HasUnboundVariability() == false);
    return type; 
}


static const Type *
lDeconstifyType(const Type *t) {
    const PointerType *pt = CastType<PointerType>(t);
    if (pt != NULL)
        return new PointerType(lDeconstifyType(pt->GetBaseType()), 
                               pt->GetVariability(), false);
    else
        return t->GetAsNonConstType();
}


Expr *
TypeCastExpr::TypeCheck() {
    if (expr == NULL)
        return NULL;

    const Type *toType = type, *fromType = expr->GetType();
    if (toType == NULL || fromType == NULL)
        return NULL;

    if (toType->HasUnboundVariability() && fromType->IsUniformType()) {
        TypeCastExpr *tce = new TypeCastExpr(toType->GetAsUniformType(),
                                             expr, pos);
        return ::TypeCheck(tce);
    }
    type = toType = type->ResolveUnboundVariability(Variability::Varying);

    fromType = lDeconstifyType(fromType);
    toType = lDeconstifyType(toType);

    // Anything can be cast to void...
    if (Type::Equal(toType, AtomicType::Void))
        return this;

    if (Type::Equal(fromType, AtomicType::Void) ||
        (fromType->IsVaryingType() && toType->IsUniformType())) {
        Error(pos, "Can't type cast from type \"%s\" to type \"%s\"",
              fromType->GetString().c_str(), toType->GetString().c_str());
        return NULL;
    }

    // First some special cases that we allow only with an explicit type cast
    const PointerType *fromPtr = CastType<PointerType>(fromType);
    const PointerType *toPtr = CastType<PointerType>(toType);
    if (fromPtr != NULL && toPtr != NULL)
        // allow explicit typecasts between any two different pointer types
        return this;

    const AtomicType *fromAtomic = CastType<AtomicType>(fromType);
    const AtomicType *toAtomic = CastType<AtomicType>(toType);
    const EnumType *fromEnum = CastType<EnumType>(fromType);
    const EnumType *toEnum = CastType<EnumType>(toType);
    if ((fromAtomic || fromEnum) && (toAtomic || toEnum))
        // Allow explicit casts between all of these
        return this;

    // ptr -> int type casts
    if (fromPtr != NULL && toAtomic != NULL && toAtomic->IsIntType()) {
        bool safeCast = (toAtomic->basicType == AtomicType::TYPE_INT64 ||
                         toAtomic->basicType == AtomicType::TYPE_UINT64);
        if (g->target.is32Bit)
            safeCast |= (toAtomic->basicType == AtomicType::TYPE_INT32 ||
                         toAtomic->basicType == AtomicType::TYPE_UINT32);
        if (safeCast == false)
            Warning(pos, "Pointer type cast of type \"%s\" to integer type "
                    "\"%s\" may lose information.", 
                    fromType->GetString().c_str(), 
                    toType->GetString().c_str());
        return this;
    }

    // int -> ptr
    if (fromAtomic != NULL && fromAtomic->IsIntType() && toPtr != NULL)
        return this;

    // And otherwise see if it's one of the conversions allowed to happen
    // implicitly.
    if (CanConvertTypes(fromType, toType, "type cast expression", pos) == false)
        return NULL;

    return this;
}


Expr *
TypeCastExpr::Optimize() {
    ConstExpr *constExpr = dynamic_cast<ConstExpr *>(expr);
    if (constExpr == NULL)
        // We can't do anything if this isn't a const expr
        return this;

    const Type *toType = GetType();
    const AtomicType *toAtomic = CastType<AtomicType>(toType);
    const EnumType *toEnum = CastType<EnumType>(toType);
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


int
TypeCastExpr::EstimateCost() const {
    if (dynamic_cast<ConstExpr *>(expr) != NULL)
        return 0;

    // FIXME: return COST_TYPECAST_COMPLEX when appropriate
    return COST_TYPECAST_SIMPLE;
}


void
TypeCastExpr::Print() const {
    printf("[%s] type cast (", GetType()->GetString().c_str());
    expr->Print();
    printf(")");
    pos.Print();
}


Symbol *
TypeCastExpr::GetBaseSymbol() const {
    return expr ? expr->GetBaseSymbol() : NULL;
}


static
llvm::Constant *
lConvertPointerConstant(llvm::Constant *c, const Type *constType) {
    if (c == NULL || constType->IsUniformType())
        return c;

    // Handle conversion to int and then to vector of int or array of int
    // (for varying and soa types, respectively)
    llvm::Constant *intPtr = 
        llvm::ConstantExpr::getPtrToInt(c, LLVMTypes::PointerIntType);
    Assert(constType->IsVaryingType() || constType->IsSOAType());
    int count = constType->IsVaryingType() ? g->target.vectorWidth :
        constType->GetSOAWidth();

    std::vector<llvm::Constant *> smear;
    for (int i = 0; i < count; ++i)
        smear.push_back(intPtr);

    if (constType->IsVaryingType())
        return llvm::ConstantVector::get(smear);
    else {
        llvm::ArrayType *at =
            llvm::ArrayType::get(LLVMTypes::PointerIntType, count);
        return llvm::ConstantArray::get(at, smear);
    }
}


llvm::Constant *
TypeCastExpr::GetConstant(const Type *constType) const {
    // We don't need to worry about most the basic cases where the type
    // cast can resolve to a constant here, since the
    // TypeCastExpr::Optimize() method generally ends up doing the type
    // conversion and returning a ConstExpr, which in turn will have its
    // GetConstant() method called.  However, because ConstExpr currently
    // can't represent pointer values, we have to handle a few cases
    // related to pointers here:
    //
    // 1. Null pointer (NULL, 0) valued initializers
    // 2. Converting function types to pointer-to-function types
    // 3. And converting these from uniform to the varying/soa equivalents.
    //
    if (CastType<PointerType>(constType) == NULL)
        return NULL;

    llvm::Constant *c = expr->GetConstant(constType->GetAsUniformType());
    return lConvertPointerConstant(c, constType);
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
    if (expr == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }
    
    llvm::Value *value = expr->GetLValue(ctx);
    if (value != NULL)
        return value;

    // value is NULL if the expression is a temporary; in this case, we'll
    // allocate storage for it so that we can return the pointer to that...
    const Type *type;
    llvm::Type *llvmType;
    if ((type = expr->GetType()) == NULL ||
        (llvmType = type->LLVMType(g->ctx)) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    value = expr->GetValue(ctx);
    if (value == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    llvm::Value *ptr = ctx->AllocaInst(llvmType);
    ctx->StoreInst(value, ptr);
    return ptr;
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

    return new ReferenceType(type);
}


const Type *
ReferenceExpr::GetLValueType() const {
    if (!expr) 
        return NULL;

    const Type *type = expr->GetType();
    if (!type) 
        return NULL;

    return PointerType::GetUniform(type);
}


Expr *
ReferenceExpr::Optimize() {
    if (expr == NULL)
        return NULL;
    return this;
}


Expr *
ReferenceExpr::TypeCheck() {
    if (expr == NULL)
        return NULL;
    return this;
}


int
ReferenceExpr::EstimateCost() const {
    return 0;
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
// DerefExpr

DerefExpr::DerefExpr(Expr *e, SourcePos p)
    : Expr(p) {
    expr = e;
}


llvm::Value *
DerefExpr::GetValue(FunctionEmitContext *ctx) const {
    if (expr == NULL) 
        return NULL;
    llvm::Value *ptr = expr->GetValue(ctx);
    if (ptr == NULL)
        return NULL;
    const Type *type = expr->GetType();
    if (type == NULL)
        return NULL;

    if (lVaryingStructHasUniformMember(type, pos))
        return NULL;

    Symbol *baseSym = expr->GetBaseSymbol();
    llvm::Value *mask = baseSym ? lMaskForSymbol(baseSym, ctx) : 
        ctx->GetFullMask();

    ctx->SetDebugPos(pos);
    return ctx->LoadInst(ptr, mask, type);
}


llvm::Value *
DerefExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (expr == NULL) 
        return NULL;
    return expr->GetValue(ctx);
}


const Type *
DerefExpr::GetLValueType() const {
    if (expr == NULL)
        return NULL;
    return expr->GetType();
}


Symbol *
DerefExpr::GetBaseSymbol() const {
    return expr ? expr->GetBaseSymbol() : NULL;
}


Expr *
DerefExpr::Optimize() {
    if (expr == NULL)
        return NULL;
    return this;
}


///////////////////////////////////////////////////////////////////////////
// PtrDerefExpr

PtrDerefExpr::PtrDerefExpr(Expr *e, SourcePos p)
    : DerefExpr(e, p) {
}


const Type *
PtrDerefExpr::GetType() const {
    const Type *type;
    if (expr == NULL || (type = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }
    AssertPos(pos, CastType<PointerType>(type) != NULL);

    if (type->IsUniformType())
        return type->GetBaseType();
    else
        return type->GetBaseType()->GetAsVaryingType();
}


Expr *
PtrDerefExpr::TypeCheck() {
    const Type *type;
    if (expr == NULL || (type = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    if (const PointerType *pt = CastType<PointerType>(type)) {
        if (Type::Equal(AtomicType::Void, pt->GetBaseType())) {
            Error(pos, "Illegal to dereference void pointer type \"%s\".",
                  type->GetString().c_str());
            return NULL;
        }
    }
    else {
        Error(pos, "Illegal to dereference non-pointer type \"%s\".", 
              type->GetString().c_str());
        return NULL;
    }

    return this;
}


int
PtrDerefExpr::EstimateCost() const {
    const Type *type;
    if (expr == NULL || (type = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return 0;
    }

    if (type->IsVaryingType())
        // Be pessimistic; some of these will later be optimized into
        // vector loads/stores..
        return COST_GATHER + COST_DEREF;
    else
        return COST_DEREF;
}


void
PtrDerefExpr::Print() const {
    if (expr == NULL || GetType() == NULL)
        return;

    printf("[%s] *(", GetType()->GetString().c_str());
    expr->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// RefDerefExpr

RefDerefExpr::RefDerefExpr(Expr *e, SourcePos p)
    : DerefExpr(e, p) {
}


const Type *
RefDerefExpr::GetType() const {
    const Type *type;
    if (expr == NULL || (type = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }
        
    AssertPos(pos, CastType<ReferenceType>(type) != NULL);
    return type->GetReferenceTarget();
}


Expr *
RefDerefExpr::TypeCheck() {
    const Type *type;
    if (expr == NULL || (type = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    // We only create RefDerefExprs internally for references in
    // expressions, so we should never create one with a non-reference
    // expression...
    AssertPos(pos, CastType<ReferenceType>(type) != NULL);

    return this;
}


int
RefDerefExpr::EstimateCost() const {
    if (expr == NULL)
        return 0;

    return COST_DEREF;
}


void
RefDerefExpr::Print() const {
    if (expr == NULL || GetType() == NULL)
        return;

    printf("[%s] deref-reference (", GetType()->GetString().c_str());
    expr->Print();
    printf(")");
    pos.Print();
}


///////////////////////////////////////////////////////////////////////////
// AddressOfExpr

AddressOfExpr::AddressOfExpr(Expr *e, SourcePos p)
    : Expr(p), expr(e) {
}


llvm::Value *
AddressOfExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    if (expr == NULL)
        return NULL;

    const Type *exprType = expr->GetType();
    if (CastType<ReferenceType>(exprType) != NULL ||
        CastType<FunctionType>(exprType) != NULL)
        return expr->GetValue(ctx);
    else
        return expr->GetLValue(ctx);
}


const Type *
AddressOfExpr::GetType() const {
    if (expr == NULL)
        return NULL;

    const Type *exprType = expr->GetType();
    if (CastType<ReferenceType>(exprType) != NULL)
        return PointerType::GetUniform(exprType->GetReferenceTarget());

    const Type *t = expr->GetLValueType();
    if (t != NULL)
        return t;
    else {
        t = expr->GetType();
        if (t == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }
        return PointerType::GetUniform(t);
    }
}


Symbol *
AddressOfExpr::GetBaseSymbol() const {
    return expr ? expr->GetBaseSymbol() : NULL;
}


void
AddressOfExpr::Print() const {
    printf("&(");
    if (expr)
        expr->Print();
    else
        printf("NULL expr");
    printf(")");
    pos.Print();
}


Expr *
AddressOfExpr::TypeCheck() {
    const Type *exprType;
    if (expr == NULL || (exprType = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    if (CastType<ReferenceType>(exprType) != NULL ||
        CastType<FunctionType>(exprType) != NULL) {
        return this;
    }

    if (expr->GetLValueType() != NULL)
        return this;

    Error(expr->pos, "Illegal to take address of non-lvalue or function.");
    return NULL;
}


Expr *
AddressOfExpr::Optimize() {
    return this;
}


int
AddressOfExpr::EstimateCost() const {
    return 0;
}


llvm::Constant *
AddressOfExpr::GetConstant(const Type *type) const {
    const Type *exprType;
    if (expr == NULL || (exprType = expr->GetType()) == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    const PointerType *pt = CastType<PointerType>(type);
    if (pt == NULL)
        return NULL;

    const FunctionType *ft = CastType<FunctionType>(pt->GetBaseType());
    if (ft != NULL) {
        llvm::Constant *c = expr->GetConstant(ft);
        return lConvertPointerConstant(c, type);
    }
    else
        return NULL;
}


///////////////////////////////////////////////////////////////////////////
// SizeOfExpr

SizeOfExpr::SizeOfExpr(Expr *e, SourcePos p) 
    : Expr(p), expr(e), type(NULL) {
}


SizeOfExpr::SizeOfExpr(const Type *t, SourcePos p)
    : Expr(p), expr(NULL), type(t) {
    type = type->ResolveUnboundVariability(Variability::Varying);
}


llvm::Value *
SizeOfExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    const Type *t = expr ? expr->GetType() : type;
    if (t == NULL)
        return NULL;

    llvm::Type *llvmType = t->LLVMType(g->ctx);
    if (llvmType == NULL)
        return NULL;

    return g->target.SizeOf(llvmType, ctx->GetCurrentBasicBlock());
}


const Type *
SizeOfExpr::GetType() const {
    return (g->target.is32Bit || g->opt.force32BitAddressing) ? 
        AtomicType::UniformUInt32 : AtomicType::UniformUInt64;
}


void
SizeOfExpr::Print() const {
    printf("Sizeof (");
    if (expr != NULL) 
        expr->Print();
    const Type *t = expr ? expr->GetType() : type;
    if (t != NULL)
        printf(" [type %s]", t->GetString().c_str());
    printf(")");
    pos.Print();
}


Expr *
SizeOfExpr::TypeCheck() {
    // Can't compute the size of a struct without a definition
    if (type != NULL &&
        CastType<UndefinedStructType>(type) != NULL) {
        Error(pos, "Can't compute the size of declared but not defined "
              "struct type \"%s\".", type->GetString().c_str());
        return NULL;
    }

    return this;
}


Expr *
SizeOfExpr::Optimize() {
    return this;
}


int
SizeOfExpr::EstimateCost() const {
    return 0;
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

    std::string loadName = symbol->name + std::string("_load");
    return ctx->LoadInst(symbol->storagePtr, loadName.c_str());
}


llvm::Value *
SymbolExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (symbol == NULL)
        return NULL;
    ctx->SetDebugPos(pos);
    return symbol->storagePtr;
}


const Type *
SymbolExpr::GetLValueType() const {
    if (symbol == NULL)
        return NULL;

    if (CastType<ReferenceType>(symbol->type) != NULL)
        return PointerType::GetUniform(symbol->type->GetReferenceTarget());
    else
        return PointerType::GetUniform(symbol->type);
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
        AssertPos(pos, GetType()->IsConstType());
        return new ConstExpr(symbol->constValue, pos);
    }
    else
        return this;
}


int
SymbolExpr::EstimateCost() const {
    // Be optimistic and assume it's in a register or can be used as a
    // memory operand..
    return 0;
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

FunctionSymbolExpr::FunctionSymbolExpr(const char *n,
                                       const std::vector<Symbol *> &candidates,
                                       SourcePos p) 
  : Expr(p) {
    name = n;
    candidateFunctions = candidates;
    matchingFunc = (candidates.size() == 1) ? candidates[0] : NULL;
    triedToResolve = false;
}


const Type *
FunctionSymbolExpr::GetType() const {
    if (triedToResolve == false && matchingFunc == NULL) {
        Error(pos, "Ambiguous use of overloaded function \"%s\".", 
              name.c_str());
        return NULL;
    }

    return matchingFunc ? matchingFunc->type : NULL;
}


llvm::Value *
FunctionSymbolExpr::GetValue(FunctionEmitContext *ctx) const {
    return matchingFunc ? matchingFunc->function : NULL;
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


int
FunctionSymbolExpr::EstimateCost() const {
    return 0;
}


void
FunctionSymbolExpr::Print() const {
    if (!matchingFunc || !GetType())
        return;

    printf("[%s] fun sym (%s)", GetType()->GetString().c_str(),
           matchingFunc->name.c_str());
    pos.Print();
}


llvm::Constant *
FunctionSymbolExpr::GetConstant(const Type *type) const {
    if (matchingFunc == NULL || matchingFunc->function == NULL)
        return NULL;

    const FunctionType *ft = CastType<FunctionType>(type);
    if (ft == NULL)
        return NULL;

    if (Type::Equal(type, matchingFunc->type) == false) {
        Error(pos, "Type of function symbol \"%s\" doesn't match expected type "
              "\"%s\".", matchingFunc->type->GetString().c_str(),
              type->GetString().c_str());
        return NULL;
    }

    return matchingFunc->function;
}


static std::string
lGetOverloadCandidateMessage(const std::vector<Symbol *> &funcs, 
                             const std::vector<const Type *> &argTypes, 
                             const std::vector<bool> *argCouldBeNULL) {
    std::string message = "Passed types: (";
    for (unsigned int i = 0; i < argTypes.size(); ++i) {
        if (argTypes[i] != NULL)
            message += argTypes[i]->GetString();
        else
            message += "(unknown type)";
        message += (i < argTypes.size()-1) ? ", " : ")\n";
    }

    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ft = CastType<FunctionType>(funcs[i]->type);
        Assert(ft != NULL);
        message += "Candidate: ";
        message += ft->GetString();
        if (i < funcs.size() - 1)
            message += "\n";
    }
    return message;
}


static bool
lIsMatchToNonConstReference(const Type *callType, const Type *funcArgType) {
    return (CastType<ReferenceType>(funcArgType) &&
            (funcArgType->IsConstType() == false) &&
            Type::Equal(callType, funcArgType->GetReferenceTarget()));
}


static bool
lIsMatchToNonConstReferenceUnifToVarying(const Type *callType,
                                         const Type *funcArgType) {
    return (CastType<ReferenceType>(funcArgType) &&
            (funcArgType->IsConstType() == false) &&
            Type::Equal(callType->GetAsVaryingType(),
                        funcArgType->GetReferenceTarget()));
}

/** Helper function used for function overload resolution: returns true if
    converting the argument to the call type only requires a type
    conversion that won't lose information.  Otherwise return false.
  */
static bool
lIsMatchWithTypeWidening(const Type *callType, const Type *funcArgType) {
    const AtomicType *callAt = CastType<AtomicType>(callType);
    const AtomicType *funcAt = CastType<AtomicType>(funcArgType);
    if (callAt == NULL || funcAt == NULL)
        return false;

    if (callAt->IsUniformType() != funcAt->IsUniformType())
        return false;

    switch (callAt->basicType) {
    case AtomicType::TYPE_BOOL:
        return true;
    case AtomicType::TYPE_INT8:
    case AtomicType::TYPE_UINT8:
        return (funcAt->basicType != AtomicType::TYPE_BOOL);
    case AtomicType::TYPE_INT16:
    case AtomicType::TYPE_UINT16:
        return (funcAt->basicType != AtomicType::TYPE_BOOL &&
                funcAt->basicType != AtomicType::TYPE_INT8 &&
                funcAt->basicType != AtomicType::TYPE_UINT8);
    case AtomicType::TYPE_INT32:
    case AtomicType::TYPE_UINT32:
        return (funcAt->basicType == AtomicType::TYPE_INT32 ||
                funcAt->basicType == AtomicType::TYPE_UINT32 ||
                funcAt->basicType == AtomicType::TYPE_INT64 ||
                funcAt->basicType == AtomicType::TYPE_UINT64);
    case AtomicType::TYPE_FLOAT:
        return (funcAt->basicType == AtomicType::TYPE_DOUBLE);
    case AtomicType::TYPE_INT64:
    case AtomicType::TYPE_UINT64:
        return (funcAt->basicType == AtomicType::TYPE_INT64 ||
                funcAt->basicType == AtomicType::TYPE_UINT64);
    case AtomicType::TYPE_DOUBLE:
        return false;
    default:
        FATAL("Unhandled atomic type");
        return false;
    }
}


/** Helper function used for function overload resolution: returns true if
    the call argument type and the function argument type match if we only
    do a uniform -> varying type conversion but otherwise have exactly the
    same type.
 */
static bool
lIsMatchWithUniformToVarying(const Type *callType, const Type *funcArgType) {
    return (callType->IsUniformType() && 
            funcArgType->IsVaryingType() &&
            Type::EqualIgnoringConst(callType->GetAsVaryingType(), funcArgType));
}


/** Helper function used for function overload resolution: returns true if
    we can type convert from the call argument type to the function
    argument type, but without doing a uniform -> varying conversion.
 */
static bool
lIsMatchWithTypeConvSameVariability(const Type *callType,
                                    const Type *funcArgType) {
    return (CanConvertTypes(callType, funcArgType) &&
            (callType->GetVariability() == funcArgType->GetVariability()));
}


/* Returns the set of function overloads that are potential matches, given
   argCount values being passed as arguments to the function call.
 */
std::vector<Symbol *>
FunctionSymbolExpr::getCandidateFunctions(int argCount) const {
    std::vector<Symbol *> ret;
    for (int i = 0; i < (int)candidateFunctions.size(); ++i) {
        const FunctionType *ft = 
            CastType<FunctionType>(candidateFunctions[i]->type);
        AssertPos(pos, ft != NULL);

        // There's no way to match if the caller is passing more arguments
        // than this function instance takes.
        if (argCount > ft->GetNumParameters())
            continue;

        // Not enough arguments, and no default argument value to save us
        if (argCount < ft->GetNumParameters() &&
            ft->GetParameterDefault(argCount) == NULL)
            continue;

        // Success
        ret.push_back(candidateFunctions[i]);
    }
    return ret;
}


static bool
lArgIsPointerType(const Type *type) {
    if (CastType<PointerType>(type) != NULL)
        return true;

    const ReferenceType *rt = CastType<ReferenceType>(type);
    if (rt == NULL)
        return false;

    const Type *t = rt->GetReferenceTarget();
    return (CastType<PointerType>(t) != NULL);
}


/** This function computes the value of a cost function that represents the
    cost of calling a function of the given type with arguments of the
    given types.  If it's not possible to call the function, regardless of
    any type conversions applied, a cost of -1 is returned.
 */
int
FunctionSymbolExpr::computeOverloadCost(const FunctionType *ftype,
                                        const std::vector<const Type *> &argTypes,
                                        const std::vector<bool> *argCouldBeNULL,
                                        const std::vector<bool> *argIsConstant) {
    int costSum = 0;

    // In computing the cost function, we only worry about the actual
    // argument types--using function default parameter values is free for
    // the purposes here...
    for (int i = 0; i < (int)argTypes.size(); ++i) {
        // The cost imposed by this argument will be a multiple of
        // costScale, which has a value set so that for each of the cost
        // buckets, even if all of the function arguments undergo the next
        // lower-cost conversion, the sum of their costs will be less than
        // a single instance of the next higher-cost conversion.
        int costScale = argTypes.size() + 1;

        const Type *fargType = ftype->GetParameterType(i);
        const Type *callType = argTypes[i];

        if (Type::Equal(callType, fargType))
            // Perfect match: no cost
            costSum += 0;
        else if (argCouldBeNULL && (*argCouldBeNULL)[i] &&
                 lArgIsPointerType(fargType))
            // Passing NULL to a pointer-typed parameter is also a no-cost
            // operation
            costSum += 0;
        else {
            // If the argument is a compile-time constant, we'd like to
            // count the cost of various conversions as much lower than the
            // cost if it wasn't--so scale up the cost when this isn't the
            // case..
            if (argIsConstant == NULL || (*argIsConstant)[i] == false)
                costScale *= 128;

            // For convenience, normalize to non-const types (except for
            // references, where const-ness matters).  For all other types,
            // we're passing by value anyway, so const doesn't matter.
            const Type *callTypeNC = callType, *fargTypeNC = fargType;
            if (CastType<ReferenceType>(callType) == NULL)
                callTypeNC = callType->GetAsNonConstType();
            if (CastType<ReferenceType>(fargType) == NULL)
                fargTypeNC = fargType->GetAsNonConstType();
                
            if (Type::Equal(callTypeNC, fargTypeNC))
                // Exact match (after dealing with references, above)
                costSum += 1 * costScale;
            // note: orig fargType for the next two...
            else if (lIsMatchToNonConstReference(callTypeNC, fargType))
                costSum += 2 * costScale;
            else if (lIsMatchToNonConstReferenceUnifToVarying(callTypeNC, fargType))
                costSum += 4 * costScale;
            else if (lIsMatchWithTypeWidening(callTypeNC, fargTypeNC))
                costSum += 8 * costScale;
            else if (lIsMatchWithUniformToVarying(callTypeNC, fargTypeNC))
                costSum += 16 * costScale;
            else if (lIsMatchWithTypeConvSameVariability(callTypeNC, fargTypeNC))
                costSum += 32 * costScale;
            else if (CanConvertTypes(callTypeNC, fargTypeNC))
                costSum += 64 * costScale;
            else
                // Failure--no type conversion possible...
                return -1;
        }
    }

    return costSum;
}


bool
FunctionSymbolExpr::ResolveOverloads(SourcePos argPos,
                                     const std::vector<const Type *> &argTypes,
                                     const std::vector<bool> *argCouldBeNULL,
                                     const std::vector<bool> *argIsConstant) {
    const char *funName = candidateFunctions.front()->name.c_str();

    triedToResolve = true;

    // Functions with names that start with "__" should only be various
    // builtins.  For those, we'll demand an exact match, since we'll
    // expect whichever function in stdlib.ispc is calling out to one of
    // those to be matching the argument types exactly; this is to be a bit
    // extra safe to be sure that the expected builtin is in fact being
    // called.
    bool exactMatchOnly = (name.substr(0,2) == "__");

    // First, find the subset of overload candidates that take the same
    // number of arguments as have parameters (including functions that
    // take more arguments but have defaults starting no later than after
    // our last parameter).
    std::vector<Symbol *> actualCandidates = 
        getCandidateFunctions(argTypes.size());

    int bestMatchCost = 1<<30;
    std::vector<Symbol *> matches;
    std::vector<int> candidateCosts;

    if (actualCandidates.size() == 0)
        goto failure;

    // Compute the cost for calling each of the candidate functions
    for (int i = 0; i < (int)actualCandidates.size(); ++i) {
        const FunctionType *ft = 
            CastType<FunctionType>(actualCandidates[i]->type);
        AssertPos(pos, ft != NULL);
        candidateCosts.push_back(computeOverloadCost(ft, argTypes,
                                                     argCouldBeNULL,
                                                     argIsConstant));
    }

    // Find the best cost, and then the candidate or candidates that have
    // that cost.
    for (int i = 0; i < (int)candidateCosts.size(); ++i) {
        if (candidateCosts[i] != -1 && candidateCosts[i] < bestMatchCost)
            bestMatchCost = candidateCosts[i];
    }
    // None of the candidates matched
    if (bestMatchCost == (1<<30))
        goto failure;
    for (int i = 0; i < (int)candidateCosts.size(); ++i) {
        if (candidateCosts[i] == bestMatchCost)
            matches.push_back(actualCandidates[i]);
    }

    if (matches.size() == 1) {
        // Only one match: success
        matchingFunc = matches[0];
        return true;
    }
    else if (matches.size() > 1) {
        // Multiple matches: ambiguous
        std::string candidateMessage = 
            lGetOverloadCandidateMessage(matches, argTypes, argCouldBeNULL);
        Error(pos, "Multiple overloaded functions matched call to function "
              "\"%s\"%s.\n%s", funName, 
              exactMatchOnly ? " only considering exact matches" : "",
              candidateMessage.c_str());
        return false;
    }
    else {
        // No matches at all
 failure:
        std::string candidateMessage = 
            lGetOverloadCandidateMessage(matches, argTypes, argCouldBeNULL);
        Error(pos, "Unable to find any matching overload for call to function "
              "\"%s\"%s.\n%s", funName, 
              exactMatchOnly ? " only considering exact matches" : "",
              candidateMessage.c_str());
        return false;
    }
}


Symbol *
FunctionSymbolExpr::GetMatchingFunction() {
    return matchingFunc;
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
    ctx->SyncInst();
    return NULL;
}


int
SyncExpr::EstimateCost() const {
    return COST_SYNC;
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


///////////////////////////////////////////////////////////////////////////
// NullPointerExpr

llvm::Value *
NullPointerExpr::GetValue(FunctionEmitContext *ctx) const {
    return llvm::ConstantPointerNull::get(LLVMTypes::VoidPointerType);
}


const Type *
NullPointerExpr::GetType() const {
    return PointerType::Void;
}


Expr *
NullPointerExpr::TypeCheck() {
    return this;
}


Expr *
NullPointerExpr::Optimize() {
    return this;
}


llvm::Constant *
NullPointerExpr::GetConstant(const Type *type) const {
    const PointerType *pt = CastType<PointerType>(type);
    if (pt == NULL)
        return NULL;

    llvm::Type *llvmType = type->LLVMType(g->ctx);
    if (llvmType == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }

    return llvm::Constant::getNullValue(llvmType);
}


void
NullPointerExpr::Print() const {
    printf("NULL");
    pos.Print();
}


int
NullPointerExpr::EstimateCost() const {
    return 0;
}


///////////////////////////////////////////////////////////////////////////
// NewExpr

NewExpr::NewExpr(int typeQual, const Type *t, Expr *init, Expr *count, 
                 SourcePos tqPos, SourcePos p)
    : Expr(p) {
    allocType = t;

    initExpr = init;
    countExpr = count;

    /* (The below cases actually should be impossible, since the parser
       doesn't allow more than a single type qualifier before a "new".) */
    if ((typeQual & ~(TYPEQUAL_UNIFORM | TYPEQUAL_VARYING)) != 0) {
        Error(tqPos, "Illegal type qualifiers in \"new\" expression (only "
              "\"uniform\" and \"varying\" are allowed.");
        isVarying = false;
    }
    else if ((typeQual & TYPEQUAL_UNIFORM) != 0 &&
             (typeQual & TYPEQUAL_VARYING) != 0) {
        Error(tqPos, "Illegal to provide both \"uniform\" and \"varying\" "
              "qualifiers to \"new\" expression.");
        isVarying = false;
    }
    else
        // If no type qualifier is given before the 'new', treat it as a
        // varying new.
        isVarying = (typeQual == 0) || (typeQual & TYPEQUAL_VARYING);

    if (allocType != NULL)
        allocType = allocType->ResolveUnboundVariability(Variability::Uniform);
}


llvm::Value *
NewExpr::GetValue(FunctionEmitContext *ctx) const {
    bool do32Bit = (g->target.is32Bit || g->opt.force32BitAddressing);

    // Determine how many elements we need to allocate.  Note that this
    // will be a varying value if this is a varying new.
    llvm::Value *countValue;
    if (countExpr != NULL) {
        countValue = countExpr->GetValue(ctx);
        if (countValue == NULL) {
            AssertPos(pos, m->errorCount > 0);
            return NULL;
        }
    }
    else {
        if (isVarying) {
            if (do32Bit) countValue = LLVMInt32Vector(1);
            else         countValue = LLVMInt64Vector(1);
        }
        else {
            if (do32Bit) countValue = LLVMInt32(1);
            else         countValue = LLVMInt64(1);
        }
    }

    // Compute the total amount of memory to allocate, allocSize, as the
    // product of the number of elements to allocate and the size of a
    // single element.
    llvm::Value *eltSize = g->target.SizeOf(allocType->LLVMType(g->ctx), 
                                            ctx->GetCurrentBasicBlock());
    if (isVarying)
        eltSize = ctx->SmearUniform(eltSize, "smear_size");
    llvm::Value *allocSize = ctx->BinaryOperator(llvm::Instruction::Mul, countValue,
                                                 eltSize, "alloc_size");

    // Determine which allocation builtin function to call: uniform or
    // varying, and taking 32-bit or 64-bit allocation counts.
    llvm::Function *func;
    if (isVarying) {
        if (do32Bit)
            func = m->module->getFunction("__new_varying32");
        else
            func = m->module->getFunction("__new_varying64");
    }
    else {
        if (allocSize->getType() != LLVMTypes::Int64Type)
            allocSize = ctx->SExtInst(allocSize, LLVMTypes::Int64Type,
                                      "alloc_size64");
        func = m->module->getFunction("__new_uniform");
    }
    AssertPos(pos, func != NULL);

    // Make the call for the the actual allocation.
    llvm::Value *ptrValue = ctx->CallInst(func, NULL, allocSize, "new");

    // Now handle initializers and returning the right type for the result.
    const Type *retType = GetType();
    if (retType == NULL)
        return NULL;
    if (isVarying) {
        if (g->target.is32Bit)
            // Convert i64 vector values to i32 if we are compiling to a
            // 32-bit target.
            ptrValue = ctx->TruncInst(ptrValue, LLVMTypes::VoidPointerVectorType,
                                      "ptr_to_32bit");

        if (initExpr != NULL) {
            // If we have an initializer expression, emit code that checks
            // to see if each lane is active and if so, runs the code to do
            // the initialization.  Note that we're we're taking advantage
            // of the fact that the __new_varying*() functions are
            // implemented to return NULL for program instances that aren't
            // executing; more generally, we should be using the current
            // execution mask for this...
            for (int i = 0; i < g->target.vectorWidth; ++i) {
                llvm::BasicBlock *bbInit = ctx->CreateBasicBlock("init_ptr");
                llvm::BasicBlock *bbSkip = ctx->CreateBasicBlock("skip_init");
                llvm::Value *p = ctx->ExtractInst(ptrValue, i);
                llvm::Value *nullValue = g->target.is32Bit ? LLVMInt32(0) :
                    LLVMInt64(0);
                // Is the pointer for the current lane non-zero?
                llvm::Value *nonNull = ctx->CmpInst(llvm::Instruction::ICmp,
                                                    llvm::CmpInst::ICMP_NE,
                                                    p, nullValue, "non_null");
                ctx->BranchInst(bbInit, bbSkip, nonNull);

                // Initialize the memory pointed to by the pointer for the
                // current lane.
                ctx->SetCurrentBasicBlock(bbInit);
                llvm::Type *ptrType = 
                    retType->GetAsUniformType()->LLVMType(g->ctx);
                llvm::Value *ptr = ctx->IntToPtrInst(p, ptrType);
                InitSymbol(ptr, allocType, initExpr, ctx, pos);
                ctx->BranchInst(bbSkip);

                ctx->SetCurrentBasicBlock(bbSkip);
            }
        }

        return ptrValue;
    }
    else {
        // For uniform news, we just need to cast the void * to be a
        // pointer of the return type and to run the code for initializers,
        // if present.
        llvm::Type *ptrType = retType->LLVMType(g->ctx);
        ptrValue = ctx->BitCastInst(ptrValue, ptrType, 
                                    LLVMGetName(ptrValue, "_cast_ptr"));

        if (initExpr != NULL)
            InitSymbol(ptrValue, allocType, initExpr, ctx, pos);

        return ptrValue;
    }
}


const Type *
NewExpr::GetType() const {
    if (allocType == NULL)
        return NULL;

    return isVarying ? PointerType::GetVarying(allocType) :
        PointerType::GetUniform(allocType);
}


Expr *
NewExpr::TypeCheck() {
    // It's illegal to call new with an undefined struct type
    if (allocType == NULL) {
        AssertPos(pos, m->errorCount > 0);
        return NULL;
    }
    if (CastType<UndefinedStructType>(allocType) != NULL) {
        Error(pos, "Can't dynamically allocate storage for declared "
              "but not defined type \"%s\".", allocType->GetString().c_str());
        return NULL;
    }

    // Otherwise we only need to make sure that if we have an expression
    // giving a number of elements to allocate that it can be converted to
    // an integer of the appropriate variability.
    if (countExpr == NULL)
        return this;

    const Type *countType;
    if ((countType = countExpr->GetType()) == NULL)
        return NULL;

    if (isVarying == false && countType->IsVaryingType()) {
        Error(pos, "Illegal to provide \"varying\" allocation count with "
              "\"uniform new\" expression.");
        return NULL;
    }

    // Figure out the type that the allocation count should be
    const Type *t = (g->target.is32Bit || g->opt.force32BitAddressing) ?
        AtomicType::UniformUInt32 : AtomicType::UniformUInt64;
    if (isVarying)
        t = t->GetAsVaryingType();

    countExpr = TypeConvertExpr(countExpr, t, "item count");
    if (countExpr == NULL)
        return NULL;

    return this;
}


Expr *
NewExpr::Optimize() {
    return this;
}


void
NewExpr::Print() const {
    printf("new (%s)", allocType ? allocType->GetString().c_str() : "NULL");
}


int
NewExpr::EstimateCost() const {
    return COST_NEW;
}
