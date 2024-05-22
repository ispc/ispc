/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file expr.cpp
    @brief Implementations of expression classes
*/

#include "expr.h"
#include "ast.h"
#include "builtins-decl.h"
#include "ctx.h"
#include "func.h"
#include "llvmutil.h"
#include "module.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#ifndef _MSC_VER
#include <inttypes.h>
#endif
#ifndef PRId64
#define PRId64 "lld"
#endif
#ifndef PRIu64
#define PRIu64 "llu"
#endif

#include <algorithm>
#include <list>
#include <set>
#include <sstream>
#include <stdio.h>

#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

using namespace ispc;

/////////////////////////////////////////////////////////////////////////////////////
// Expr

llvm::Value *Expr::GetLValue(FunctionEmitContext *ctx) const {
    // Expressions that can't provide an lvalue can just return nullptr
    return nullptr;
}

const Type *Expr::GetLValueType() const {
    // This also only needs to be overrided by Exprs that implement the
    // GetLValue() method.
    return nullptr;
}

std::pair<llvm::Constant *, bool> Expr::GetStorageConstant(const Type *type) const { return GetConstant(type); }
std::pair<llvm::Constant *, bool> Expr::GetConstant(const Type *type) const {
    // The default is failure; just return nullptr
    return std::pair<llvm::Constant *, bool>(nullptr, false);
}

Symbol *Expr::GetBaseSymbol() const {
    // Not all expressions can do this, so provide a generally-useful
    // default implementation.
    return nullptr;
}

bool Expr::HasAmbiguousVariability(std::vector<const Expr *> &warn) const { return false; }

///////////////////////////////////////////////////////////////////////////

static llvm::APFloat lCreateAPFloat(llvm::APFloat f, llvm::Type *type) {
    const llvm::fltSemantics &FS = type->getFltSemantics();
    bool ignored;
    f.convert(FS, llvm::APFloat::rmNearestTiesToEven, &ignored);
    return f;
}

static llvm::APFloat lCreateAPFloat(double value, llvm::Type *type) {
    llvm::APFloat f(value);
    const llvm::fltSemantics &FS = type->getFltSemantics();
    bool ignored;
    f.convert(FS, llvm::APFloat::rmNearestTiesToEven, &ignored);
    return f;
}

static Expr *lArrayToPointer(Expr *expr) {
    Assert(expr != nullptr);
    AssertPos(expr->pos, CastType<ArrayType>(expr->GetType()));

    Expr *zero = new ConstExpr(AtomicType::UniformInt32, 0, expr->pos);
    Expr *index = new IndexExpr(expr, zero, expr->pos);
    Expr *addr = new AddressOfExpr(index, expr->pos);
    addr = TypeCheck(addr);
    Assert(addr != nullptr);
    addr = Optimize(addr);
    Assert(addr != nullptr);
    return addr;
}

static bool lIsAllIntZeros(Expr *expr) {
    const Type *type = expr->GetType();
    if (type == nullptr || type->IsIntType() == false)
        return false;

    ConstExpr *ce = llvm::dyn_cast<ConstExpr>(expr);
    if (ce == nullptr)
        return false;

    uint64_t vals[ISPC_MAX_NVEC];
    int count = ce->GetValues(vals);
    if (count == 1)
        return (vals[0] == 0);
    else {
        for (int i = 0; i < count; ++i)
            if (vals[i] != 0)
                return false;
    }
    return true;
}

static bool lDoTypeConv(const Type *fromType, const Type *toType, Expr **expr, bool failureOk, const char *errorMsgBase,
                        SourcePos pos) {
    /* This function is way too long and complex.  Is type conversion stuff
       always this messy, or can this be cleaned up somehow? */
    AssertPos(pos, failureOk || errorMsgBase != nullptr);

    if (toType == nullptr || fromType == nullptr)
        return false;

    // The types are equal; there's nothing to do
    if (Type::Equal(toType, fromType))
        return true;

    if (fromType->IsVoidType()) {
        if (!failureOk)
            Error(pos, "Can't convert from \"void\" to \"%s\" for %s.", toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if (toType->IsVoidType()) {
        if (!failureOk)
            Error(pos, "Can't convert type \"%s\" to \"void\" for %s.", fromType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if (CastType<FunctionType>(fromType)) {
        if (CastType<PointerType>(toType) != nullptr) {
            // Convert function type to pointer to function type
            if (expr != nullptr) {
                Expr *aoe = new AddressOfExpr(*expr, (*expr)->pos);
                if (lDoTypeConv(aoe->GetType(), toType, &aoe, failureOk, errorMsgBase, pos)) {
                    *expr = aoe;
                    return true;
                }
            } else
                return lDoTypeConv(PointerType::GetUniform(fromType), toType, nullptr, failureOk, errorMsgBase, pos);
        } else {
            if (!failureOk)
                Error(pos, "Can't convert function type \"%s\" to \"%s\" for %s.", fromType->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return false;
        }
    }
    if (CastType<FunctionType>(toType)) {
        if (!failureOk)
            Error(pos,
                  "Can't convert from type \"%s\" to function type \"%s\" "
                  "for %s.",
                  fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if ((toType->GetSOAWidth() > 0 || fromType->GetSOAWidth() > 0) &&
        Type::Equal(toType->GetAsUniformType(), fromType->GetAsUniformType()) &&
        toType->GetSOAWidth() != fromType->GetSOAWidth()) {
        if (!failureOk)
            Error(pos,
                  "Can't convert between types \"%s\" and \"%s\" with "
                  "different SOA widths for %s.",
                  fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
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
    if (fromArrayType != nullptr && toPointerType != nullptr) {
        // can convert any array to a void pointer (both uniform and
        // varying).
        if (PointerType::IsVoidPointer(toPointerType))
            goto typecast_ok;

        // array to pointer to array element type
        const Type *eltType = fromArrayType->GetElementType();
        if (toPointerType->GetBaseType()->IsConstType())
            eltType = eltType->GetAsConstType();

        PointerType pt(eltType, toPointerType->GetVariability(), toPointerType->IsConstType());
        if (Type::Equal(toPointerType, &pt))
            goto typecast_ok;
        else {
            if (!failureOk)
                Error(pos,
                      "Can't convert from incompatible array type \"%s\" "
                      "to pointer type \"%s\" for %s.",
                      fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
            return false;
        }
    }

    if (toType->IsUniformType() && fromType->IsVaryingType()) {
        if (!failureOk)
            Error(pos, "Can't convert from type \"%s\" to type \"%s\" for %s.", fromType->GetString().c_str(),
                  toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if (fromPointerType != nullptr) {
        if (CastType<AtomicType>(toType) != nullptr && toType->IsBoolType())
            // Allow implicit conversion of pointers to bools
            goto typecast_ok;

        if (toArrayType != nullptr && Type::Equal(fromType->GetBaseType(), toArrayType->GetElementType())) {
            // Can convert pointers to arrays of the same type
            goto typecast_ok;
        }
        if (toPointerType == nullptr) {
            if (!failureOk)
                Error(pos,
                      "Can't convert between from pointer type "
                      "\"%s\" to non-pointer type \"%s\" for %s.",
                      fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
            return false;
        } else if (fromPointerType->IsSlice() == true && toPointerType->IsSlice() == false) {
            if (!failureOk)
                Error(pos,
                      "Can't convert from pointer to SOA type "
                      "\"%s\" to pointer to non-SOA type \"%s\" for %s.",
                      fromPointerType->GetAsNonSlice()->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
            return false;
        } else if (PointerType::IsVoidPointer(toPointerType)) {
            if (fromPointerType->GetBaseType()->IsConstType() && !(toPointerType->GetBaseType()->IsConstType())) {
                if (!failureOk)
                    Error(pos, "Can't convert pointer to const \"%s\" to void pointer.",
                          fromPointerType->GetString().c_str());
                return false;
            }
            // any pointer type can be converted to a void *
            // ...almost. #731
            goto typecast_ok;
        } else if (PointerType::IsVoidPointer(fromPointerType) && expr != nullptr &&
                   llvm::dyn_cast<NullPointerExpr>(*expr) != nullptr) {
            // and a nullptr convert to any other pointer type
            goto typecast_ok;
        } else if (!Type::Equal(fromPointerType->GetBaseType(), toPointerType->GetBaseType()) &&
                   !Type::Equal(fromPointerType->GetBaseType()->GetAsConstType(), toPointerType->GetBaseType())) {
            // for const * -> * conversion, print warning.
            if (Type::EqualIgnoringConst(fromPointerType->GetBaseType(), toPointerType->GetBaseType())) {
                if (!Type::Equal(fromPointerType->GetBaseType()->GetAsConstType(), toPointerType->GetBaseType())) {
                    Warning(pos,
                            "Converting from const pointer type \"%s\" to "
                            "pointer type \"%s\" for %s discards const qualifier.",
                            fromPointerType->GetString().c_str(), toPointerType->GetString().c_str(), errorMsgBase);
                }
            } else {
                if (!failureOk) {
                    Error(pos,
                          "Can't convert from pointer type \"%s\" to "
                          "incompatible pointer type \"%s\" for %s.",
                          fromPointerType->GetString().c_str(), toPointerType->GetString().c_str(), errorMsgBase);
                }
                return false;
            }
        }

        if (toType->IsVaryingType() && fromType->IsUniformType())
            goto typecast_ok;

        if (toPointerType->IsSlice() == true && fromPointerType->IsSlice() == false)
            goto typecast_ok;

        // Otherwise there's nothing to do
        return true;
    }

    if (toPointerType != nullptr && fromAtomicType != nullptr && fromAtomicType->IsIntType() && expr != nullptr &&
        lIsAllIntZeros(*expr)) {
        // We have a zero-valued integer expression, which can also be
        // treated as a nullptr pointer that can be converted to any other
        // pointer type.
        Expr *npe = new NullPointerExpr(pos);
        if (lDoTypeConv(PointerType::Void, toType, &npe, failureOk, errorMsgBase, pos)) {
            *expr = npe;
            return true;
        }
        return false;
    }

    // Need to check this early, since otherwise the [sic] "unbound"
    // variability of SOA struct types causes things to get messy if that
    // hasn't been detected...
    if (toStructType && fromStructType && (toStructType->GetSOAWidth() != fromStructType->GetSOAWidth())) {
        if (!failureOk)
            Error(pos,
                  "Can't convert between incompatible struct types \"%s\" "
                  "and \"%s\" for %s.",
                  fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    // Convert from type T -> const T; just return a TypeCast expr, which
    // can handle this
    if (Type::EqualIgnoringConst(toType, fromType) && toType->IsConstType() == true && fromType->IsConstType() == false)
        goto typecast_ok;

    if (CastType<ReferenceType>(fromType)) {
        if (CastType<ReferenceType>(toType)) {
            // Convert from a reference to a type to a const reference to a type;
            // this is handled by TypeCastExpr
            if (Type::Equal(toType->GetReferenceTarget(), fromType->GetReferenceTarget()->GetAsConstType()))
                goto typecast_ok;

            const ArrayType *atFrom = CastType<ArrayType>(fromType->GetReferenceTarget());
            const ArrayType *atTo = CastType<ArrayType>(toType->GetReferenceTarget());

            if (atFrom != nullptr && atTo != nullptr && Type::Equal(atFrom->GetElementType(), atTo->GetElementType())) {
                goto typecast_ok;
            } else {
                if (!failureOk)
                    Error(pos,
                          "Can't convert between incompatible reference types \"%s\" "
                          "and \"%s\" for %s.",
                          fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
                return false;
            }
        } else {
            // convert from a reference T -> T
            if (expr != nullptr) {
                Expr *drExpr = new RefDerefExpr(*expr, pos);
                if (lDoTypeConv(drExpr->GetType(), toType, &drExpr, failureOk, errorMsgBase, pos) == true) {
                    *expr = drExpr;
                    return true;
                }
                return false;
            } else
                return lDoTypeConv(fromType->GetReferenceTarget(), toType, nullptr, failureOk, errorMsgBase, pos);
        }
    } else if (CastType<ReferenceType>(toType)) {
        // T -> reference T
        if (expr != nullptr) {
            Expr *rExpr = new ReferenceExpr(*expr, pos);
            if (lDoTypeConv(rExpr->GetType(), toType, &rExpr, failureOk, errorMsgBase, pos) == true) {
                *expr = rExpr;
                return true;
            }
            return false;
        } else {
            ReferenceType rt(fromType);
            return lDoTypeConv(&rt, toType, nullptr, failureOk, errorMsgBase, pos);
        }
    } else if (Type::Equal(toType, fromType->GetAsNonConstType()))
        // convert: const T -> T (as long as T isn't a reference)
        goto typecast_ok;

    fromType = fromType->GetReferenceTarget();
    toType = toType->GetReferenceTarget();
    if (toArrayType && fromArrayType) {
        if (Type::Equal(toArrayType->GetElementType(), fromArrayType->GetElementType())) {
            // the case of different element counts should have returned
            // successfully earlier, yes??
            AssertPos(pos, toArrayType->GetElementCount() != fromArrayType->GetElementCount());
            goto typecast_ok;
        } else if (Type::Equal(toArrayType->GetElementType(), fromArrayType->GetElementType()->GetAsConstType())) {
            // T[x] -> const T[x]
            goto typecast_ok;
        } else {
            if (!failureOk)
                Error(pos, "Array type \"%s\" can't be converted to type \"%s\" for %s.", fromType->GetString().c_str(),
                      toType->GetString().c_str(), errorMsgBase);
            return false;
        }
    }

    if (toVectorType && fromVectorType) {
        // converting e.g. int<n> -> float<n>
        if (fromVectorType->GetElementCount() != toVectorType->GetElementCount()) {
            if (!failureOk)
                Error(pos,
                      "Can't convert between differently sized vector types "
                      "\"%s\" -> \"%s\" for %s.",
                      fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
            return false;
        }
        goto typecast_ok;
    }

    if (toStructType && fromStructType) {
        if (!Type::Equal(toStructType->GetAsUniformType()->GetAsConstType(),
                         fromStructType->GetAsUniformType()->GetAsConstType())) {
            if (!failureOk)
                Error(pos,
                      "Can't convert between different struct types "
                      "\"%s\" and \"%s\" for %s.",
                      fromStructType->GetString().c_str(), toStructType->GetString().c_str(), errorMsgBase);
            return false;
        }
        goto typecast_ok;
    }

    if (toEnumType != nullptr && fromEnumType != nullptr) {
        // No implicit conversions between different enum types
        if (!Type::EqualIgnoringConst(toEnumType->GetAsUniformType(), fromEnumType->GetAsUniformType())) {
            if (!failureOk)
                Error(pos,
                      "Can't convert between different enum types "
                      "\"%s\" and \"%s\" for %s",
                      fromEnumType->GetString().c_str(), toEnumType->GetString().c_str(), errorMsgBase);
            return false;
        }
        goto typecast_ok;
    }

    // enum -> atomic (integer, generally...) is always ok
    if (fromEnumType != nullptr) {
        // Cannot convert to anything other than atomic
        if (toAtomicType == nullptr && toVectorType == nullptr) {
            if (!failureOk)
                Error(pos,
                      "Type conversion from \"%s\" to \"%s\" for %s is not "
                      "possible.",
                      fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
            return false;
        }
        goto typecast_ok;
    }

    // from here on out, the from type can only be atomic something or
    // other...
    if (fromAtomicType == nullptr) {
        if (!failureOk)
            Error(pos,
                  "Type conversion from \"%s\" to \"%s\" for %s is not "
                  "possible.",
                  fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    // scalar -> short-vector conversions
    if (toVectorType != nullptr && (fromType->GetSOAWidth() == toType->GetSOAWidth()))
        goto typecast_ok;

    // ok, it better be a scalar->scalar conversion of some sort by now
    if (toAtomicType == nullptr) {
        if (!failureOk)
            Error(pos,
                  "Type conversion from \"%s\" to \"%s\" for %s is "
                  "not possible",
                  fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
        return false;
    }

    if (fromType->GetSOAWidth() != toType->GetSOAWidth()) {
        if (!failureOk)
            Error(pos,
                  "Can't convert between types \"%s\" and \"%s\" with "
                  "different SOA widths for %s.",
                  fromType->GetString().c_str(), toType->GetString().c_str(), errorMsgBase);
        return false;
    }

typecast_ok:
    if (expr != nullptr)
        *expr = new TypeCastExpr(toType, *expr, pos);
    return true;
}

bool ispc::CanConvertTypes(const Type *fromType, const Type *toType, const char *errorMsgBase, SourcePos pos) {
    return lDoTypeConv(fromType, toType, nullptr, errorMsgBase == nullptr, errorMsgBase, pos);
}

Expr *ispc::TypeConvertExpr(Expr *expr, const Type *toType, const char *errorMsgBase) {
    if (expr == nullptr)
        return nullptr;

    const Type *fromType = expr->GetType();
    Expr *e = expr;
    if (lDoTypeConv(fromType, toType, &e, false, errorMsgBase, expr->pos))
        return e;
    else
        return nullptr;
}

bool ispc::PossiblyResolveFunctionOverloads(Expr *expr, const Type *type) {
    FunctionSymbolExpr *fse = nullptr;
    const FunctionType *funcType = nullptr;
    if (CastType<PointerType>(type) != nullptr && (funcType = CastType<FunctionType>(type->GetBaseType())) &&
        (fse = llvm::dyn_cast<FunctionSymbolExpr>(expr)) != nullptr) {
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

static llvm::Value *lTypeConvAtomicOrUniformVector(FunctionEmitContext *ctx, llvm::Value *exprVal, const Type *toType,
                                                   const Type *fromType, SourcePos pos);
/** Utility routine that emits code to initialize a symbol given an
    initializer expression.

    @param ptr       Memory location of storage for the symbol's data
    @param symName   Name of symbol (used in error messages)
    @param symType   Type of variable being initialized
    @param initExpr  Expression for the initializer
    @param ctx       FunctionEmitContext to use for generating instructions
    @param pos       Source file position of the variable being initialized
*/
void ispc::InitSymbol(AddressInfo *ptrInfo, const Type *symType, Expr *initExpr, FunctionEmitContext *ctx,
                      SourcePos pos) {
    if (initExpr == nullptr)
        // leave it uninitialized
        return;

    // See if we have a constant initializer a this point
    std::pair<llvm::Constant *, bool> constValPair = initExpr->GetStorageConstant(symType);
    llvm::Constant *constValue = constValPair.first;
    if (constValue != nullptr) {
        // It'd be nice if we could just do a StoreInst(constValue, ptr)
        // at this point, but unfortunately that doesn't generate great
        // code (e.g. a bunch of scalar moves for a constant array.)  So
        // instead we'll make a constant static global that holds the
        // constant value and emit a memcpy to put its value into the
        // pointer we have.
        llvm::Type *llvmType = symType->LLVMStorageType(g->ctx);
        if (llvmType == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return;
        }

        if (Type::IsBasicType(symType))
            ctx->StoreInst(constValue, ptrInfo, symType);
        else {
            llvm::Value *constPtr =
                new llvm::GlobalVariable(*m->module, llvmType, true /* const */, llvm::GlobalValue::InternalLinkage,
                                         constValue, "const_initializer");
            llvm::Value *size = g->target->SizeOf(llvmType, ctx->GetCurrentBasicBlock());
            ctx->MemcpyInst(ptrInfo->getPointer(), constPtr, size);
        }

        return;
    }

    // If the initializer is a straight up expression that isn't an
    // ExprList, then we'll see if we can type convert it to the type of
    // the variable.
    if (llvm::dyn_cast<ExprList>(initExpr) == nullptr) {
        if (PossiblyResolveFunctionOverloads(initExpr, symType) == false)
            return;
        initExpr = TypeConvertExpr(initExpr, symType, "initializer");

        if (initExpr == nullptr)
            return;

        llvm::Value *initializerValue = initExpr->GetValue(ctx);
        if (initializerValue != nullptr)
            // Bingo; store the value in the variable's storage
            ctx->StoreInst(initializerValue, ptrInfo, symType);
        return;
    }

    // Atomic types and enums can be initialized with { ... } initializer
    // expressions if they have a single element (except for SOA types,
    // which are handled below).
    if (symType->IsSOAType() == false && Type::IsBasicType(symType)) {
        ExprList *elist = llvm::dyn_cast<ExprList>(initExpr);
        if (elist != nullptr) {
            if (elist->exprs.size() == 1) {
                InitSymbol(ptrInfo, symType, elist->exprs[0], ctx, pos);
                return;
            } else if (symType->IsVaryingType() == false) {
                Error(initExpr->pos,
                      "Expression list initializers with "
                      "multiple values can't be used with type \"%s\".",
                      symType->GetString().c_str());
                return;
            }
        } else
            return;
    }

    const ReferenceType *rt = CastType<ReferenceType>(symType);
    if (rt) {
        const Type *initExprType = initExpr->GetType();
        Assert(initExprType);
        if (!Type::Equal(initExprType, rt)) {
            Error(initExpr->pos,
                  "Initializer for reference type \"%s\" must have same "
                  "reference type itself. \"%s\" is incompatible.",
                  rt->GetString().c_str(), initExprType->GetString().c_str());
            return;
        }

        llvm::Value *initializerValue = initExpr->GetValue(ctx);
        if (initializerValue)
            ctx->StoreInst(initializerValue, ptrInfo, initExprType);
        return;
    }

    // Handle initiailizers for SOA types as well as for structs, arrays,
    // and vectors.
    const CollectionType *collectionType = CastType<CollectionType>(symType);
    if (collectionType != nullptr || symType->IsSOAType() ||
        (Type::IsBasicType(symType) && symType->IsVaryingType() == true)) {
        // Make default value equivalent to number of elements for varying
        int nElements = g->target->getVectorWidth();
        if (collectionType)
            nElements = collectionType->GetElementCount();
        else if (symType->IsSOAType())
            nElements = symType->GetSOAWidth();

        std::string name;
        if (CastType<StructType>(symType) != nullptr)
            name = "struct";
        else if (CastType<ArrayType>(symType) != nullptr)
            name = "array";
        else if (CastType<VectorType>(symType) != nullptr)
            name = "vector";
        else if (symType->IsSOAType() || (Type::IsBasicType(symType) && symType->IsVaryingType() == true))
            name = symType->GetVariability().GetString();
        else
            FATAL("Unexpected CollectionType in InitSymbol()");

        // There are two cases for initializing these types; either a
        // single initializer may be provided (float foo[3] = 0;), in which
        // case all of the elements are initialized to the given value, or
        // an initializer list may be provided (float foo[3] = { 1,2,3 }),
        // in which case the elements are initialized with the
        // corresponding values.
        ExprList *exprList = llvm::dyn_cast<ExprList>(initExpr);
        if (exprList != nullptr) {
            // The { ... } case; make sure we have the no more expressions
            // in the ExprList as we have struct members
            int nInits = exprList->exprs.size();
            if (nInits > nElements) {
                Error(initExpr->pos,
                      "Initializer for %s type \"%s\" requires "
                      "no more than %d values; %d provided.",
                      name.c_str(), symType->GetString().c_str(), nElements, nInits);
                return;
            } else if ((Type::IsBasicType(symType) && symType->IsVaryingType() == true) && (nInits < nElements)) {
                Error(initExpr->pos,
                      "Initializer for %s type \"%s\" requires "
                      "%d values; %d provided.",
                      name.c_str(), symType->GetString().c_str(), nElements, nInits);
                return;
            }

            // When we initialize a uniform vector, try to construct a vector from initializers and use a vector
            // instructions for type conversion if needed. It makes sense to generate vector instruction for type
            // conversion when at least two initializers are of the same atomic type. For example: const uniform
            // float<4> f = {1.f, 2.f, 3.f, 4.f}; const uniform double<4> d = {f[1], 1.d, f.y, 0.d};
            if (symType->IsVectorType() && symType->IsUniformType() && nInits > 0) {
                std::map<AtomicType::BasicType, std::vector<ExprList::ExprPosMapping>> exprPerType;
                if (exprList->HasAtomicInitializerList(exprPerType)) {
                    const VectorType *symVectorType = CastType<VectorType>(symType);
                    const int nVectorElements = symVectorType->getVectorMemoryCount();
                    // The idea is to produce the sequence of instructions that performs the following:
                    // 1. construct a vector of initializers of the same type
                    // 2. convert it to target type
                    // 3. shuffle it with resulting vector
                    // For example:
                    // uniform double<4> d = {f.x, d.x, f.y, d.y};
                    // vResVal1 = <4 x float> <f.x, undef, f.y, undef>
                    // vResVal1_conv = fpext <4 x float> <f.x, undef, f.y, undef> to <4 x double>
                    // vResVal2 = <4 x double> <undef, d.x, undef, d.y>
                    // initListVector = shufflevector %vResVal1_conv, <4 x double> %undef, <4 x i32> <i32 0, i32 5, i32
                    // 3, i32 7>
                    // initListVector = shufflevector %vResVal2, <4 x double> %initListVector, <4 x i32> <i32 4, i32 1,
                    // i32 6, i32 3>

                    // Our final vector which will be used for initialization
                    llvm::Value *initListVector = llvm::UndefValue::get(
                        llvm::FixedVectorType::get(symVectorType->GetElementType()->LLVMType(g->ctx), nVectorElements));
                    for (const auto &[type, expr_map] : exprPerType) {
                        // There is no good way to construct AtomicType from AtomicType::BasicType,
                        // just use the first one
                        Assert(expr_map.size() > 0);
                        Expr *expr0 = expr_map[0].expr;
                        const AtomicType *aType = CastType<AtomicType>(expr0->GetType());
                        Assert(aType);

                        // If we have two or more initializers of the same type, generate vector instruction.
                        if (expr_map.size() > 1) {
                            // Construct ISPC vector type for resulting "initializer" vector for this type
                            const VectorType *vResType = new VectorType(aType, nVectorElements);
                            // Resulting LLVM vector for this type
                            llvm::Value *initListVectorPerType = llvm::UndefValue::get(
                                llvm::FixedVectorType::get(aType->LLVMType(g->ctx), nVectorElements));
                            // Create a linear vector that will be used as a shuffle mask for
                            // shufflevector with initListVector.
                            std::vector<uint32_t> linearVector(nVectorElements);
                            std::generate(linearVector.begin(), linearVector.end(),
                                          [n = nVectorElements]() mutable { return n++; });
                            // Insert initializers of this type to initListVectorPerType
                            for (const auto &init_expr : expr_map) {
                                llvm::Value *initializerValue = init_expr.expr->GetValue(ctx);
                                initListVectorPerType =
                                    ctx->InsertInst(initListVectorPerType, initializerValue, init_expr.pos);
                                linearVector[init_expr.pos] = init_expr.pos;
                            }

                            // Make type conversion
                            if (!Type::EqualIgnoringConst(symType, vResType)) {
                                initListVectorPerType =
                                    lTypeConvAtomicOrUniformVector(ctx, initListVectorPerType, symType, vResType, pos);
                            }
                            // If we have all initializers of the same type, we don't need to generate shuffle
                            if (exprPerType.size() > 1) {
                                // Construct LLVM vector for shuffle mask
                                llvm::Value *initIndex =
                                    llvm::ConstantDataVector::get(*g->ctx, llvm::ArrayRef<uint32_t>(linearVector));
                                initListVector = ctx->ShuffleInst(initListVectorPerType, initListVector, initIndex);
                            } else {
                                initListVector = initListVectorPerType;
                            }
                        } else {
                            // Make type conversion if needed
                            llvm::Value *conv = expr0->GetValue(ctx);
                            if (!Type::EqualIgnoringConst(symVectorType->GetElementType(), aType)) {
                                conv = lTypeConvAtomicOrUniformVector(ctx, expr0->GetValue(ctx),
                                                                      symVectorType->GetElementType(), aType, pos);
                            }
                            // Insert initializer into initializer vector
                            initListVector = ctx->InsertInst(initListVector, conv, expr_map[0].pos);
                        }
                    }
                    ctx->StoreInst(initListVector, ptrInfo);
                    return;
                }
            }
            // Initialize each element with the corresponding value from
            // the ExprList
            for (int i = 0; i < nElements; ++i) {
                // For SOA types and varying, the element type is the uniform variant
                // of the underlying type
                const Type *elementType =
                    collectionType ? collectionType->GetElementType(i) : symType->GetAsUniformType();
                llvm::Value *ep;
                if (CastType<StructType>(symType) != nullptr)
                    ep = ctx->AddElementOffset(new AddressInfo(ptrInfo->getPointer(), CastType<StructType>(symType)), i,
                                               "element");
                else {
                    ep = ctx->GetElementPtrInst(ptrInfo->getPointer(), LLVMInt32(0), LLVMInt32(i),
                                                /* Type of aggregate structure */ PointerType::GetUniform(symType),
                                                "gep");
                }
                AddressInfo *epInfo = new AddressInfo(ep, ptrInfo->getElementType());
                if (i < nInits)
                    InitSymbol(epInfo, elementType, exprList->exprs[i], ctx, pos);
                else {
                    // If we don't have enough initializer values, initialize the
                    // rest as zero.
                    llvm::Type *llvmType = elementType->LLVMStorageType(g->ctx);
                    if (llvmType == nullptr) {
                        AssertPos(pos, m->errorCount > 0);
                        return;
                    }

                    llvm::Constant *zeroInit = llvm::Constant::getNullValue(llvmType);
                    ctx->StoreInst(zeroInit, epInfo, elementType);
                }
            }
        } else if (collectionType) {
            Error(initExpr->pos, "Can't assign type \"%s\" to \"%s\".", initExpr->GetType()->GetString().c_str(),
                  collectionType->GetString().c_str());
        } else {
            FATAL("CollectionType is nullptr in InitSymbol()");
        }
        return;
    }

    FATAL("Unexpected Type in InitSymbol()");
}

///////////////////////////////////////////////////////////////////////////

/** Given an atomic or vector type, this returns a boolean type with the
    same "shape".  In other words, if the given type is a vector type of
    three uniform ints, the returned type is a vector type of three uniform
    bools. */
static const Type *lMatchingBoolType(const Type *type) {
    bool uniformTest = type->IsUniformType();
    const AtomicType *boolBase = uniformTest ? AtomicType::UniformBool : AtomicType::VaryingBool;
    const VectorType *vt = CastType<VectorType>(type);
    if (vt != nullptr)
        return new VectorType(boolBase, vt->GetElementCount());
    else {
        Assert(Type::IsBasicType(type) || type->IsReferenceType());
        return boolBase;
    }
}

///////////////////////////////////////////////////////////////////////////
// UnaryExpr
static llvm::Constant *lLLVMConstantValue(const Type *type, llvm::LLVMContext *ctx, double value) {
    const AtomicType *atomicType = CastType<AtomicType>(type);
    const EnumType *enumType = CastType<EnumType>(type);
    const VectorType *vectorType = CastType<VectorType>(type);
    const PointerType *pointerType = CastType<PointerType>(type);

    // This function is only called with, and only works for atomic, enum,
    // and vector types.
    Assert(atomicType != nullptr || enumType != nullptr || vectorType != nullptr || pointerType != nullptr);

    if (atomicType != nullptr || enumType != nullptr) {
        // If it's an atomic or enuemrator type, then figure out which of
        // the llvmutil.h functions to call to get the corresponding
        // constant and then call it...
        bool isUniform = type->IsUniformType();
        AtomicType::BasicType basicType = (enumType != nullptr) ? AtomicType::TYPE_UINT32 : atomicType->basicType;

        switch (basicType) {
        case AtomicType::TYPE_VOID:
            FATAL("can't get constant value for void type");
            return nullptr;
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
        case AtomicType::TYPE_INT64: {
            int64_t i = (int64_t)value;
            Assert((double)i == value);
            return isUniform ? LLVMInt64(i) : LLVMInt64Vector(i);
        }
        case AtomicType::TYPE_UINT64: {
            uint64_t i = (uint64_t)value;
            Assert(value == (int64_t)i);
            return isUniform ? LLVMUInt64(i) : LLVMUInt64Vector(i);
        }
        case AtomicType::TYPE_FLOAT16: {
            llvm::APFloat apf16 = lCreateAPFloat(value, LLVMTypes::Float16Type);
            return isUniform ? LLVMFloat16(apf16) : LLVMFloat16Vector(apf16);
        }
        case AtomicType::TYPE_FLOAT: {
            llvm::APFloat apf = lCreateAPFloat(value, LLVMTypes::FloatType);
            return isUniform ? LLVMFloat(apf) : LLVMFloatVector(apf);
        }
        case AtomicType::TYPE_DOUBLE: {
            llvm::APFloat apd = lCreateAPFloat(value, LLVMTypes::DoubleType);
            return isUniform ? LLVMDouble(apd) : LLVMDoubleVector(apd);
        }
        default:
            FATAL("logic error in lLLVMConstantValue");
            return nullptr;
        }
    } else if (pointerType != nullptr) {
        Assert(value == 0);
        if (pointerType->IsUniformType())
            return llvm::Constant::getNullValue(LLVMTypes::VoidPointerType);
        else
            return llvm::Constant::getNullValue(LLVMTypes::VoidPointerVectorType);
    } else {
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
            llvm::FixedVectorType *lvt = llvm::dyn_cast<llvm::FixedVectorType>(llvmVectorType);
            Assert(lvt != nullptr);
            std::vector<llvm::Constant *> vals;
            for (unsigned int i = 0; i < lvt->getNumElements(); ++i)
                vals.push_back(constElement);
            return llvm::ConstantVector::get(vals);
        } else {
            llvm::ArrayType *lat = llvm::dyn_cast<llvm::ArrayType>(llvmVectorType);
            Assert(lat != nullptr);
            std::vector<llvm::Constant *> vals;
            for (unsigned int i = 0; i < lat->getNumElements(); ++i)
                vals.push_back(constElement);
            return llvm::ConstantArray::get(lat, vals);
        }
    }
}

static llvm::Value *lMaskForSymbol(Symbol *baseSym, FunctionEmitContext *ctx) {
    if (baseSym == nullptr)
        return ctx->GetFullMask();

    if (CastType<PointerType>(baseSym->type) != nullptr || CastType<ReferenceType>(baseSym->type) != nullptr)
        // FIXME: for pointers, we really only want to do this for
        // dereferencing the pointer, not for things like pointer
        // arithmetic, when we may be able to use the internal mask,
        // depending on context...
        return ctx->GetFullMask();

    llvm::Value *mask = (baseSym->parentFunction == ctx->GetFunction() && baseSym->storageClass != SC_STATIC)
                            ? ctx->GetInternalMask()
                            : ctx->GetFullMask();
    return mask;
}

/** Store the result of an assignment to the given location.
 */
static void lStoreAssignResult(llvm::Value *value, llvm::Value *ptr, const Type *valueType, const Type *ptrType,
                               FunctionEmitContext *ctx, Symbol *baseSym) {
    Assert(baseSym == nullptr || baseSym->varyingCFDepth <= ctx->VaryingCFDepth());
    if (!g->opt.disableMaskedStoreToStore && !g->opt.disableMaskAllOnOptimizations && baseSym != nullptr &&
        baseSym->varyingCFDepth == ctx->VaryingCFDepth() && baseSym->storageClass != SC_STATIC &&
        CastType<ReferenceType>(baseSym->type) == nullptr && CastType<PointerType>(baseSym->type) == nullptr) {
        // If the variable is declared at the same varying control flow
        // depth as where it's being assigned, then we don't need to do any
        // masking but can just do the assignment as if all the lanes were
        // known to be on.  While this may lead to random/garbage values
        // written into the lanes that are off, by definition they will
        // never be accessed, since those lanes aren't executing, and won't
        // be executing at this scope or any other one before the variable
        // goes out of scope.
        ctx->StoreInst(value, ptr, LLVMMaskAllOn, valueType, ptrType);
    } else {
        ctx->StoreInst(value, ptr, lMaskForSymbol(baseSym, ctx), valueType, ptrType);
    }
}

/** Utility routine to emit code to do a {pre,post}-{inc,dec}rement of the
    given expresion.
 */
static llvm::Value *lEmitPrePostIncDec(UnaryExpr::Op op, Expr *expr, SourcePos pos, FunctionEmitContext *ctx,
                                       WrapSemantics wrapSemantics) {
    const Type *type = expr->GetType();
    if (type == nullptr)
        return nullptr;

    // Get both the lvalue and the rvalue of the given expression
    llvm::Value *lvalue = nullptr, *rvalue = nullptr;
    const Type *lvalueType = nullptr;
    if (CastType<ReferenceType>(type) != nullptr) {
        lvalueType = type;
        type = type->GetReferenceTarget();
        lvalue = expr->GetValue(ctx);

        RefDerefExpr *deref = new RefDerefExpr(expr, expr->pos);
        rvalue = deref->GetValue(ctx);
    } else {
        lvalue = expr->GetLValue(ctx);
        lvalueType = expr->GetLValueType();
        rvalue = expr->GetValue(ctx);
    }

    if (lvalue == nullptr) {
        // If we can't get a lvalue, then we have an error here
        const char *prepost = (op == UnaryExpr::PreInc || op == UnaryExpr::PreDec) ? "pre" : "post";
        const char *incdec = (op == UnaryExpr::PreInc || op == UnaryExpr::PostInc) ? "increment" : "decrement";
        Error(pos, "Can't %s-%s non-lvalues.", prepost, incdec);
        return nullptr;
    }

    // Emit code to do the appropriate addition/subtraction to the
    // expression's old value
    ctx->SetDebugPos(pos);
    llvm::Value *binop = nullptr;
    int delta = (op == UnaryExpr::PreInc || op == UnaryExpr::PostInc) ? 1 : -1;

    std::string opName = rvalue->getName().str();
    if (op == UnaryExpr::PreInc || op == UnaryExpr::PostInc)
        opName += "_plus1";
    else
        opName += "_minus1";

    if (CastType<PointerType>(type) != nullptr) {
        const Type *incType = type->IsUniformType() ? AtomicType::UniformInt32 : AtomicType::VaryingInt32;
        llvm::Constant *dval = lLLVMConstantValue(incType, g->ctx, delta);
        binop = ctx->GetElementPtrInst(rvalue, dval, type, opName.c_str());
    } else {
        llvm::Constant *dval = lLLVMConstantValue(type, g->ctx, delta);
        if (type->IsFloatType())
            binop = ctx->BinaryOperator(llvm::Instruction::FAdd, rvalue, dval, WrapSemantics::None, opName.c_str());
        else
            binop = ctx->BinaryOperator(llvm::Instruction::Add, rvalue, dval, wrapSemantics, opName.c_str());
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
static llvm::Value *lEmitNegate(Expr *arg, SourcePos pos, FunctionEmitContext *ctx, WrapSemantics wrapSemantics) {
    const Type *type = arg->GetType();
    llvm::Value *argVal = arg->GetValue(ctx);
    if (type == nullptr || argVal == nullptr)
        return nullptr;

    // Negate by subtracting from zero...
    ctx->SetDebugPos(pos);
    if (type->IsFloatType()) {
        return ctx->FNegInst(argVal, llvm::Twine(argVal->getName()) + "_negate");
    } else {
        llvm::Value *zero = lLLVMConstantValue(type, g->ctx, 0.);
        AssertPos(pos, type->IsIntType());
        return ctx->BinaryOperator(llvm::Instruction::Sub, zero, argVal, wrapSemantics,
                                   llvm::Twine(argVal->getName()) + "_negate");
    }
}

UnaryExpr::UnaryExpr(Op o, Expr *e, SourcePos p) : Expr(p, UnaryExprID), op(o) { expr = e; }

llvm::Value *UnaryExpr::GetValue(FunctionEmitContext *ctx) const {
    if (expr == nullptr)
        return nullptr;

    const auto wrapSemantics = this->GetType()->IsSignedType() ? WrapSemantics::NSW : WrapSemantics::None;
    ctx->SetDebugPos(pos);

    switch (op) {
    case PreInc:
    case PreDec:
    case PostInc:
    case PostDec:
        return lEmitPrePostIncDec(op, expr, pos, ctx, wrapSemantics);
    case Negate:
        return lEmitNegate(expr, pos, ctx, wrapSemantics);
    case LogicalNot: {
        llvm::Value *argVal = expr->GetValue(ctx);
        return ctx->NotOperator(argVal, llvm::Twine(argVal->getName()) + "_logicalnot");
    }
    case BitNot: {
        llvm::Value *argVal = expr->GetValue(ctx);
        return ctx->NotOperator(argVal, llvm::Twine(argVal->getName()) + "_bitnot");
    }
    default:
        FATAL("logic error");
        return nullptr;
    }
}

const Type *UnaryExpr::GetType() const {
    if (expr == nullptr)
        return nullptr;

    const Type *type = expr->GetType();
    if (type == nullptr)
        return nullptr;

    if (type->IsDependentType()) {
        return AtomicType::Dependent;
    }

    // Unary expressions should be returning target types after updating
    // reference address.
    if (CastType<ReferenceType>(type) != nullptr) {
        type = type->GetReferenceTarget();
    }
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
        return nullptr;
    }
}

template <typename T> static Expr *lOptimizeBitNot(ConstExpr *constExpr, const Type *type, SourcePos pos) {
    T v[ISPC_MAX_NVEC];
    int count = constExpr->GetValues(v);
    for (int i = 0; i < count; ++i)
        v[i] = ~v[i];
    return new ConstExpr(type, v, pos);
}

template <typename T> static Expr *lOptimizeNegate(ConstExpr *constExpr, const Type *type, SourcePos pos) {
    T v[ISPC_MAX_NVEC];
    int count = constExpr->GetValues(v);
    for (int i = 0; i < count; ++i)
        v[i] = -v[i];
    return new ConstExpr(type, v, pos);
}

Expr *UnaryExpr::Optimize() {
    ConstExpr *constExpr = llvm::dyn_cast<ConstExpr>(expr);
    // If the operand isn't a constant, then we can't do any optimization
    // here...
    if (constExpr == nullptr)
        return this;

    const Type *type = constExpr->GetType();
    bool isEnumType = CastType<EnumType>(type) != nullptr;

    switch (op) {
    case PreInc:
    case PreDec:
    case PostInc:
    case PostDec:
        // this shouldn't happen--it's illegal to modify a contant value..
        // An error will be issued elsewhere...
        return this;
    case Negate: {
        if (Type::EqualIgnoringConst(type, AtomicType::UniformInt64) ||
            Type::EqualIgnoringConst(type, AtomicType::VaryingInt64)) {
            return lOptimizeNegate<int64_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt64) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt64)) {
            return lOptimizeNegate<uint64_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformInt32) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingInt32)) {
            return lOptimizeNegate<int32_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt32) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt32)) {
            return lOptimizeNegate<uint32_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformInt16) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingInt16)) {
            return lOptimizeNegate<int16_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt16) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt32)) {
            return lOptimizeNegate<uint16_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformInt8) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingInt8)) {
            return lOptimizeNegate<int8_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt8) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt8)) {
            return lOptimizeNegate<uint8_t>(constExpr, type, pos);
        } else {
            AssertPos(pos, Type::EqualIgnoringConst(type, AtomicType::UniformFloat16) ||
                               Type::EqualIgnoringConst(type, AtomicType::VaryingFloat16) ||
                               Type::EqualIgnoringConst(type, AtomicType::UniformFloat) ||
                               Type::EqualIgnoringConst(type, AtomicType::VaryingFloat) ||
                               Type::EqualIgnoringConst(type, AtomicType::UniformDouble) ||
                               Type::EqualIgnoringConst(type, AtomicType::VaryingDouble));
            std::vector<llvm::APFloat> v;
            int count = constExpr->GetValues(v);
            for (int i = 0; i < count; ++i)
                v[i].changeSign();
            return new ConstExpr(type, v, pos);
        }
    }
    case BitNot: {
        if (Type::EqualIgnoringConst(type, AtomicType::UniformInt8) ||
            Type::EqualIgnoringConst(type, AtomicType::VaryingInt8)) {
            return lOptimizeBitNot<int8_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt8) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt8)) {
            return lOptimizeBitNot<uint8_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformInt16) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingInt16)) {
            return lOptimizeBitNot<int16_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt16) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt16)) {
            return lOptimizeBitNot<uint16_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformInt32) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingInt32)) {
            return lOptimizeBitNot<int32_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt32) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt32) || isEnumType == true) {
            return lOptimizeBitNot<uint32_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformInt64) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingInt64)) {
            return lOptimizeBitNot<int64_t>(constExpr, type, pos);
        } else if (Type::EqualIgnoringConst(type, AtomicType::UniformUInt64) ||
                   Type::EqualIgnoringConst(type, AtomicType::VaryingUInt64)) {
            return lOptimizeBitNot<uint64_t>(constExpr, type, pos);
        } else
            FATAL("unexpected type in UnaryExpr::Optimize() / BitNot case");
    }
    case LogicalNot: {
        AssertPos(pos, Type::EqualIgnoringConst(type, AtomicType::UniformBool) ||
                           Type::EqualIgnoringConst(type, AtomicType::VaryingBool));
        bool v[ISPC_MAX_NVEC];
        int count = constExpr->GetValues(v);
        for (int i = 0; i < count; ++i)
            v[i] = !v[i];
        return new ConstExpr(type, v, pos);
    }
    default:
        FATAL("unexpected op in UnaryExpr::Optimize()");
        return nullptr;
    }
}

Expr *UnaryExpr::TypeCheck() {
    const Type *type;
    if (expr == nullptr || (type = expr->GetType()) == nullptr)
        // something went wrong in type checking...
        return nullptr;

    if (type->IsDependentType()) {
        return this;
    }

    if (type->IsSOAType()) {
        Error(pos, "Can't apply unary operator to SOA type \"%s\".", type->GetString().c_str());
        return nullptr;
    }

    if (op == PreInc || op == PreDec || op == PostInc || op == PostDec) {
        if (type->IsConstType()) {
            Error(pos,
                  "Can't assign to type \"%s\" on left-hand side of "
                  "expression.",
                  type->GetString().c_str());
            return nullptr;
        }

        if (type->IsNumericType())
            return this;

        const PointerType *pt = CastType<PointerType>(type);
        if (pt == nullptr) {
            Error(expr->pos,
                  "Can only pre/post increment numeric and "
                  "pointer types, not \"%s\".",
                  type->GetString().c_str());
            return nullptr;
        }

        if (PointerType::IsVoidPointer(type)) {
            Error(expr->pos, "Illegal to pre/post increment \"%s\" type.", type->GetString().c_str());
            return nullptr;
        }
        if (CastType<UndefinedStructType>(pt->GetBaseType())) {
            Error(expr->pos,
                  "Illegal to pre/post increment pointer to "
                  "undefined struct type \"%s\".",
                  type->GetString().c_str());
            return nullptr;
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
            Error(expr->pos, "Negate not allowed for non-numeric type \"%s\".", type->GetString().c_str());
            return nullptr;
        }
    } else if (op == LogicalNot) {
        const Type *boolType = lMatchingBoolType(type);
        expr = TypeConvertExpr(expr, boolType, "logical not");
        if (expr == nullptr)
            return nullptr;
    } else if (op == BitNot) {
        if (!type->IsIntType()) {
            Error(expr->pos,
                  "~ operator can only be used with integer types, "
                  "not \"%s\".",
                  type->GetString().c_str());
            return nullptr;
        }
    }
    return this;
}

int UnaryExpr::EstimateCost() const {
    if (llvm::dyn_cast<ConstExpr>(expr) != nullptr)
        return 0;

    return COST_SIMPLE_ARITH_LOGIC_OP;
}

UnaryExpr *UnaryExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return new UnaryExpr(op, instExpr, pos);
}

void UnaryExpr::Print(Indent &indent) const {
    if (!expr || !GetType()) {
        indent.Print("UnaryExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("UnaryExpr", pos);

    printf("[ %s ] ", GetType()->GetString().c_str());
    switch (op) {
    case PreInc: ///< Pre-increment
        printf("prefix '++'");
        break;
    case PreDec: ///< Pre-decrement
        printf("prefix '--'");
        break;
    case PostInc: ///< Post-increment
        printf("postfix '++'");
        break;
    case PostDec: ///< Post-decrement
        printf("postfix '--'");
        break;
    case Negate: ///< Negation
        printf("prefix '-'");
        break;
    case LogicalNot: ///< Logical not
        printf("prefix '!'");
        break;
    case BitNot:
        printf("prefix '~'");
        break;
    default:
        printf("<ILLEGAL OP");
        break;
    }
    printf("\n");
    indent.pushSingle();
    expr->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// BinaryExpr

static const char *lOpString(BinaryExpr::Op op) {
    switch (op) {
    case BinaryExpr::Add:
        return "+";
    case BinaryExpr::Sub:
        return "-";
    case BinaryExpr::Mul:
        return "*";
    case BinaryExpr::Div:
        return "/";
    case BinaryExpr::Mod:
        return "%";
    case BinaryExpr::Shl:
        return "<<";
    case BinaryExpr::Shr:
        return ">>";
    case BinaryExpr::Lt:
        return "<";
    case BinaryExpr::Gt:
        return ">";
    case BinaryExpr::Le:
        return "<=";
    case BinaryExpr::Ge:
        return ">=";
    case BinaryExpr::Equal:
        return "==";
    case BinaryExpr::NotEqual:
        return "!=";
    case BinaryExpr::BitAnd:
        return "&";
    case BinaryExpr::BitXor:
        return "^";
    case BinaryExpr::BitOr:
        return "|";
    case BinaryExpr::LogicalAnd:
        return "&&";
    case BinaryExpr::LogicalOr:
        return "||";
    case BinaryExpr::Comma:
        return ",";
    default:
        FATAL("unimplemented case in lOpString()");
        return "";
    }
}

/** Utility routine to emit the binary bitwise operator corresponding to
    the given BinaryExpr::Op.
*/
static llvm::Value *lEmitBinaryBitOp(BinaryExpr::Op op, llvm::Value *arg0Val, llvm::Value *arg1Val, bool isUnsigned,
                                     FunctionEmitContext *ctx) {
    llvm::Instruction::BinaryOps inst;
    switch (op) {
    case BinaryExpr::Shl:
        inst = llvm::Instruction::Shl;
        break;
    case BinaryExpr::Shr:
        if (isUnsigned)
            inst = llvm::Instruction::LShr;
        else
            inst = llvm::Instruction::AShr;
        break;
    case BinaryExpr::BitAnd:
        inst = llvm::Instruction::And;
        break;
    case BinaryExpr::BitXor:
        inst = llvm::Instruction::Xor;
        break;
    case BinaryExpr::BitOr:
        inst = llvm::Instruction::Or;
        break;
    default:
        FATAL("logic error in lEmitBinaryBitOp()");
        return nullptr;
    }

    return ctx->BinaryOperator(inst, arg0Val, arg1Val, WrapSemantics::None, "bitop");
}

static llvm::Value *lEmitBinaryPointerArith(BinaryExpr::Op op, llvm::Value *value0, llvm::Value *value1,
                                            const Type *type0, const Type *type1, FunctionEmitContext *ctx,
                                            SourcePos pos) {
    const PointerType *ptrType = CastType<PointerType>(type0);
    AssertPos(pos, ptrType != nullptr);
    switch (op) {
    case BinaryExpr::Add:
        // ptr + integer
        return ctx->GetElementPtrInst(value0, value1, ptrType, "ptrmath");
    case BinaryExpr::Sub: {
        if (CastType<PointerType>(type1) != nullptr) {
            AssertPos(pos, Type::EqualIgnoringConst(type0, type1));

            if (ptrType->IsSlice()) {
                llvm::Value *p0 = ctx->ExtractInst(value0, 0);
                llvm::Value *p1 = ctx->ExtractInst(value1, 0);
                const Type *majorType = ptrType->GetAsNonSlice();
                llvm::Value *majorDelta = lEmitBinaryPointerArith(op, p0, p1, majorType, majorType, ctx, pos);

                int soaWidth = ptrType->GetBaseType()->GetSOAWidth();
                AssertPos(pos, soaWidth > 0);
                llvm::Value *soaScale = LLVMIntAsType(soaWidth, majorDelta->getType());

                llvm::Value *majorScale = ctx->BinaryOperator(llvm::Instruction::Mul, majorDelta, soaScale,
                                                              WrapSemantics::None, "major_soa_scaled");

                llvm::Value *m0 = ctx->ExtractInst(value0, 1);
                llvm::Value *m1 = ctx->ExtractInst(value1, 1);
                llvm::Value *minorDelta =
                    ctx->BinaryOperator(llvm::Instruction::Sub, m0, m1, WrapSemantics::None, "minor_soa_delta");

                ctx->MatchIntegerTypes(&majorScale, &minorDelta);
                return ctx->BinaryOperator(llvm::Instruction::Add, majorScale, minorDelta, WrapSemantics::None,
                                           "soa_ptrdiff");
            }

            // ptr - ptr
            if (ptrType->IsUniformType()) {
                value0 = ctx->PtrToIntInst(value0);
                value1 = ctx->PtrToIntInst(value1);
            }

            // Compute the difference in bytes
            llvm::Value *delta =
                ctx->BinaryOperator(llvm::Instruction::Sub, value0, value1, WrapSemantics::None, "ptr_diff");

            // Now divide by the size of the type that the pointer
            // points to in order to return the difference in elements.
            llvm::Type *llvmElementType = ptrType->GetBaseType()->LLVMType(g->ctx);
            llvm::Value *size = g->target->SizeOf(llvmElementType, ctx->GetCurrentBasicBlock());
            if (ptrType->IsVaryingType())
                size = ctx->SmearUniform(size);

            if (g->target->is32Bit() == false && g->opt.force32BitAddressing == true) {
                // If we're doing 32-bit addressing math on a 64-bit
                // target, then trunc the delta down to a 32-bit value.
                // (Thus also matching what will be a 32-bit value
                // returned from SizeOf above.)
                if (ptrType->IsUniformType())
                    delta = ctx->TruncInst(delta, LLVMTypes::Int32Type, "trunc_ptr_delta");
                else
                    delta = ctx->TruncInst(delta, LLVMTypes::Int32VectorType, "trunc_ptr_delta");
            }

            // And now do the actual division
            return ctx->BinaryOperator(llvm::Instruction::SDiv, delta, size, WrapSemantics::None, "element_diff");
        } else {
            // ptr - integer
            llvm::Value *zero = lLLVMConstantValue(type1, g->ctx, 0.);
            llvm::Value *negOffset =
                ctx->BinaryOperator(llvm::Instruction::Sub, zero, value1, WrapSemantics::None, "negate");
            // Do a GEP as ptr + -integer
            return ctx->GetElementPtrInst(value0, negOffset, ptrType, "ptrmath");
        }
    }
    default:
        FATAL("Logic error in lEmitBinaryArith() for pointer type case");
        return nullptr;
    }
}

/** Utility routine to emit binary arithmetic operator based on the given
    BinaryExpr::Op.
*/
static llvm::Value *lEmitBinaryArith(BinaryExpr::Op op, llvm::Value *value0, llvm::Value *value1, const Type *type0,
                                     const Type *type1, FunctionEmitContext *ctx, SourcePos pos,
                                     WrapSemantics wrapSemantics) {
    const PointerType *ptrType = CastType<PointerType>(type0);

    if (ptrType != nullptr)
        return lEmitBinaryPointerArith(op, value0, value1, type0, type1, ctx, pos);
    else {
        AssertPos(pos, Type::EqualIgnoringConst(type0, type1));

        llvm::Instruction::BinaryOps inst;
        bool isFloatOp = type0->IsFloatType();
        bool isUnsignedOp = type0->IsUnsignedType();

        const char *opName = nullptr;
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
            if (type0->IsVaryingType() && !isFloatOp && g->target->shouldWarn(PerfWarningType::DIVModInt)) {
                PerformanceWarning(pos, "Division with varying integer types is very inefficient.");
            }
            inst = isFloatOp ? llvm::Instruction::FDiv
                             : (isUnsignedOp ? llvm::Instruction::UDiv : llvm::Instruction::SDiv);
            break;
        case BinaryExpr::Mod:
            opName = "mod";
            if (type0->IsVaryingType() && !isFloatOp && g->target->shouldWarn(PerfWarningType::DIVModInt)) {
                PerformanceWarning(pos, "Modulus operator with varying types is very inefficient.");
            }
            inst = isFloatOp ? llvm::Instruction::FRem
                             : (isUnsignedOp ? llvm::Instruction::URem : llvm::Instruction::SRem);
            break;
        default:
            FATAL("Invalid op type passed to lEmitBinaryArith()");
            return nullptr;
        }

        return ctx->BinaryOperator(inst, value0, value1, wrapSemantics,
                                   (((llvm::Twine(opName) + "_") + value0->getName()) + "_") + value1->getName());
    }
}

/** Utility routine to emit a binary comparison operator based on the given
    BinaryExpr::Op.
 */
static llvm::Value *lEmitBinaryCmp(BinaryExpr::Op op, llvm::Value *e0Val, llvm::Value *e1Val, const Type *type,
                                   FunctionEmitContext *ctx, SourcePos pos) {
    bool isFloatOp = type->IsFloatType();
    bool isUnsignedOp = type->IsUnsignedType();

    llvm::CmpInst::Predicate pred;
    const char *opName = nullptr;
    switch (op) {
    case BinaryExpr::Lt:
        opName = "less";
        pred = isFloatOp ? llvm::CmpInst::FCMP_OLT : (isUnsignedOp ? llvm::CmpInst::ICMP_ULT : llvm::CmpInst::ICMP_SLT);
        break;
    case BinaryExpr::Gt:
        opName = "greater";
        pred = isFloatOp ? llvm::CmpInst::FCMP_OGT : (isUnsignedOp ? llvm::CmpInst::ICMP_UGT : llvm::CmpInst::ICMP_SGT);
        break;
    case BinaryExpr::Le:
        opName = "lessequal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_OLE : (isUnsignedOp ? llvm::CmpInst::ICMP_ULE : llvm::CmpInst::ICMP_SLE);
        break;
    case BinaryExpr::Ge:
        opName = "greaterequal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_OGE : (isUnsignedOp ? llvm::CmpInst::ICMP_UGE : llvm::CmpInst::ICMP_SGE);
        break;
    case BinaryExpr::Equal:
        opName = "equal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_OEQ : llvm::CmpInst::ICMP_EQ;
        break;
    case BinaryExpr::NotEqual:
        opName = "notequal";
        pred = isFloatOp ? llvm::CmpInst::FCMP_UNE : llvm::CmpInst::ICMP_NE;
        break;
    default:
        FATAL("error in lEmitBinaryCmp()");
        return nullptr;
    }

    llvm::Value *cmp = ctx->CmpInst(isFloatOp ? llvm::Instruction::FCmp : llvm::Instruction::ICmp, pred, e0Val, e1Val,
                                    (((llvm::Twine(opName) + "_") + e0Val->getName()) + "_") + e1Val->getName());
    // This is a little ugly: CmpInst returns i1 values, but we use vectors
    // of i32s for varying bool values; type convert the result here if
    // needed.
    if (type->IsVaryingType())
        cmp = ctx->I1VecToBoolVec(cmp);

    return cmp;
}

BinaryExpr::BinaryExpr(Op o, Expr *a, Expr *b, SourcePos p) : Expr(p, BinaryExprID), op(o) {
    arg0 = a;
    arg1 = b;
}

bool lCreateBinaryOperatorCall(const BinaryExpr::Op bop, Expr *a0, Expr *a1, Expr *&op, const SourcePos &sp) {
    bool abort = false;
    if ((a0 == nullptr) || (a1 == nullptr)) {
        return abort;
    }
    Expr *arg0 = a0;
    Expr *arg1 = a1;
    const Type *type0 = arg0->GetType();
    const Type *type1 = arg1->GetType();

    // If either operand is a reference, dereference it before we move
    // forward
    if (CastType<ReferenceType>(type0) != nullptr) {
        arg0 = new RefDerefExpr(arg0, arg0->pos);
        type0 = arg0->GetType();
    }
    if (CastType<ReferenceType>(type1) != nullptr) {
        arg1 = new RefDerefExpr(arg1, arg1->pos);
        type1 = arg1->GetType();
    }
    if ((type0 == nullptr) || (type1 == nullptr)) {
        return abort;
    }
    if (CastType<StructType>(type0) != nullptr || CastType<StructType>(type1) != nullptr) {
        std::string opName = std::string("operator") + lOpString(bop);
        std::vector<Symbol *> funs;
        m->symbolTable->LookupFunction(opName.c_str(), &funs);
        if (funs.size() > 0) {
            Expr *func = new FunctionSymbolExpr(opName.c_str(), funs, sp);
            ExprList *args = new ExprList(sp);
            args->exprs.push_back(arg0);
            args->exprs.push_back(arg1);
            op = new FunctionCallExpr(func, args, sp);
            return abort;
        }

        // templates
        std::vector<TemplateSymbol *> funcTempls;
        bool foundAny = m->symbolTable->LookupFunctionTemplate(opName.c_str(), &funcTempls);
        if (foundAny && funcTempls.size() > 0) {
            TemplateArgs templArgs;
            FunctionSymbolExpr *functionSymbolExpr = new FunctionSymbolExpr(opName.c_str(), funcTempls, templArgs, sp);
            Assert(functionSymbolExpr != nullptr);
            ExprList *args = new ExprList(sp);
            args->exprs.push_back(arg0);
            args->exprs.push_back(arg1);
            op = new FunctionCallExpr(functionSymbolExpr, args, sp);
            return abort;
        }

        if (funs.size() == 0 && funcTempls.size() == 0) {
            Error(sp, "operator %s(%s, %s) is not defined.", opName.c_str(), (type0->GetString()).c_str(),
                  (type1->GetString()).c_str());
            abort = true;
            return abort;
        }

        return abort;
    }
    return abort;
}

Expr *ispc::MakeBinaryExpr(BinaryExpr::Op o, Expr *a, Expr *b, SourcePos p) {
    Expr *op = nullptr;
    bool abort = lCreateBinaryOperatorCall(o, a, b, op, p);
    if (op != nullptr) {
        return op;
    }

    // lCreateBinaryOperatorCall can return nullptr for 2 cases:
    // 1. When there is an error.
    // 2. We have to create a new BinaryExpr.
    if (abort) {
        AssertPos(p, m->errorCount > 0);
        return nullptr;
    }

    op = new BinaryExpr(o, a, b, p);
    return op;
}

/** Emit code for a && or || logical operator.  In particular, the code
    here handles "short-circuit" evaluation, where the second expression
    isn't evaluated if the value of the first one determines the value of
    the result.
*/
llvm::Value *lEmitLogicalOp(BinaryExpr::Op op, Expr *arg0, Expr *arg1, FunctionEmitContext *ctx, SourcePos pos) {

    const Type *type0 = arg0->GetType(), *type1 = arg1->GetType();
    if (type0 == nullptr || type1 == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    // There is overhead (branches, etc.), to short-circuiting, so if the
    // right side of the expression is a) relatively simple, and b) can be
    // safely executed with an all-off execution mask, then we just
    // evaluate both sides and then the logical operator in that case.
    int threshold =
        g->target->isXeTarget() ? PREDICATE_SAFE_SHORT_CIRC_XE_STATEMENT_COST : PREDICATE_SAFE_IF_STATEMENT_COST;
    bool shortCircuit = (EstimateCost(arg1) > threshold || SafeToRunWithMaskAllOff(arg1) == false);

    // Skip short-circuiting for VectorTypes as well.
    if ((shortCircuit == false) || CastType<VectorType>(type0) != nullptr || CastType<VectorType>(type1) != nullptr) {
        // If one of the operands is uniform but the other is varying,
        // promote the uniform one to varying
        if (type0->IsUniformType() && type1->IsVaryingType()) {
            arg0 = TypeConvertExpr(arg0, AtomicType::VaryingBool, lOpString(op));
            AssertPos(pos, arg0 != nullptr);
        }
        if (type1->IsUniformType() && type0->IsVaryingType()) {
            arg1 = TypeConvertExpr(arg1, AtomicType::VaryingBool, lOpString(op));
            AssertPos(pos, arg1 != nullptr);
        }

        llvm::Value *value0 = arg0->GetValue(ctx);
        llvm::Value *value1 = arg1->GetValue(ctx);
        if (value0 == nullptr || value1 == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }

        if (op == BinaryExpr::LogicalAnd)
            return ctx->BinaryOperator(llvm::Instruction::And, value0, value1, WrapSemantics::None, "logical_and");
        else {
            AssertPos(pos, op == BinaryExpr::LogicalOr);
            return ctx->BinaryOperator(llvm::Instruction::Or, value0, value1, WrapSemantics::None, "logical_or");
        }
    }

    // Allocate temporary storage for the return value
    const Type *retType = Type::MoreGeneralType(type0, type1, pos, lOpString(op));
    if (retType == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }
    AddressInfo *retPtrInfo = ctx->AllocaInst(retType, "logical_op_mem");
    llvm::BasicBlock *bbSkipEvalValue1 = ctx->CreateBasicBlock("skip_eval_1", ctx->GetCurrentBasicBlock());
    llvm::BasicBlock *bbEvalValue1 = ctx->CreateBasicBlock("eval_1", bbSkipEvalValue1);
    llvm::BasicBlock *bbLogicalDone = ctx->CreateBasicBlock("logical_op_done", bbEvalValue1);

    // Evaluate the first operand
    llvm::Value *value0 = arg0->GetValue(ctx);
    if (value0 == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (type0->IsUniformType()) {
        // Check to see if the value of the first operand is true or false
        llvm::Value *value0True = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, value0, LLVMTrue);

        if (op == BinaryExpr::LogicalOr) {
            // For ||, if value0 is true, then we skip evaluating value1
            // entirely.
            ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, value0True);

            // If value0 is true, the complete result is true (either
            // uniform or varying)
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            llvm::Value *trueValue = retType->IsUniformType() ? LLVMTrue : LLVMMaskAllOn;
            ctx->StoreInst(trueValue, retPtrInfo, retType);
            ctx->BranchInst(bbLogicalDone);
        } else {
            AssertPos(pos, op == BinaryExpr::LogicalAnd);

            // Conversely, for &&, if value0 is false, we skip evaluating
            // value1.
            ctx->BranchInst(bbEvalValue1, bbSkipEvalValue1, value0True);

            // In this case, the complete result is false (again, either a
            // uniform or varying false).
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            llvm::Value *falseValue = retType->IsUniformType() ? LLVMFalse : LLVMMaskAllOff;
            ctx->StoreInst(falseValue, retPtrInfo, retType);
            ctx->BranchInst(bbLogicalDone);
        }

        // Both || and && are in the same situation if the first operand's
        // value didn't resolve the final result: they need to evaluate the
        // value of the second operand, which in turn gives the value for
        // the full expression.
        ctx->SetCurrentBasicBlock(bbEvalValue1);
        if (type1->IsUniformType() && retType->IsVaryingType()) {
            arg1 = TypeConvertExpr(arg1, AtomicType::VaryingBool, "logical op");
            AssertPos(pos, arg1 != nullptr);
        }

        llvm::Value *value1 = arg1->GetValue(ctx);
        if (value1 == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }
        ctx->StoreInst(value1, retPtrInfo, arg1->GetType());
        ctx->BranchInst(bbLogicalDone);

        // In all cases, we end up at the bbLogicalDone basic block;
        // loading the value stored in retPtr in turn gives the overall
        // result.
        ctx->SetCurrentBasicBlock(bbLogicalDone);
        return ctx->LoadInst(retPtrInfo, retType);
    } else {
        // Otherwise, the first operand is varying...  Save the current
        // value of the mask so that we can restore it at the end.
        llvm::Value *oldMask = ctx->GetInternalMask();
        llvm::Value *oldFullMask = ctx->GetFullMask();

        // Convert the second operand to be varying as well, so that we can
        // perform logical vector ops with its value.
        if (type1->IsUniformType()) {
            arg1 = TypeConvertExpr(arg1, AtomicType::VaryingBool, "logical op");
            AssertPos(pos, arg1 != nullptr);
            type1 = arg1->GetType();
        }

        if (op == BinaryExpr::LogicalOr) {
            // See if value0 is true for all currently executing
            // lanes--i.e. if (value0 & mask) == mask.  If so, we don't
            // need to evaluate the second operand of the expression.
            llvm::Value *value0AndMask =
                ctx->BinaryOperator(llvm::Instruction::And, value0, oldFullMask, WrapSemantics::None, "op&mask");
            llvm::Value *equalsMask = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, value0AndMask,
                                                   oldFullMask, "value0&mask==mask");
            equalsMask = ctx->I1VecToBoolVec(equalsMask);
            if (!ctx->emitXeHardwareMask()) {
                llvm::Value *allMatch = ctx->All(equalsMask);
                ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, allMatch);
            } else {
                // If uniform CF is emulated, pass vector value to BranchInst
                ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, equalsMask);
            }

            // value0 is true for all running lanes, so it can be used for
            // the final result
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            ctx->StoreInst(value0, retPtrInfo, arg0->GetType());
            ctx->BranchInst(bbLogicalDone);

            // Otherwise, we need to valuate arg1. However, first we need
            // to set the execution mask to be (oldMask & ~a); in other
            // words, only execute the instances where value0 is false.
            // For the instances where value0 was true, we need to inhibit
            // execution.
            ctx->SetCurrentBasicBlock(bbEvalValue1);
            ctx->SetInternalMaskAndNot(oldMask, value0);

            llvm::Value *value1 = arg1->GetValue(ctx);
            if (value1 == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                return nullptr;
            }

            // We need to compute the result carefully, since vector
            // elements that were computed when the corresponding lane was
            // disabled have undefined values:
            // result = (value0 & old_mask) | (value1 & current_mask)
            llvm::Value *value1AndMask = ctx->BinaryOperator(llvm::Instruction::And, value1, ctx->GetInternalMask(),
                                                             WrapSemantics::None, "op&mask");
            llvm::Value *result = ctx->BinaryOperator(llvm::Instruction::Or, value0AndMask, value1AndMask,
                                                      WrapSemantics::None, "or_result");
            ctx->StoreInst(result, retPtrInfo, retType);
            ctx->BranchInst(bbLogicalDone);
        } else {
            AssertPos(pos, op == BinaryExpr::LogicalAnd);

            // If value0 is false for all currently running lanes, the
            // overall result must be false: this corresponds to checking
            // if (mask & ~value0) == mask.
            llvm::Value *notValue0 = ctx->NotOperator(value0, "not_value0");
            llvm::Value *notValue0AndMask = ctx->BinaryOperator(llvm::Instruction::And, notValue0, oldFullMask,
                                                                WrapSemantics::None, "not_value0&mask");
            llvm::Value *equalsMask = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_EQ, notValue0AndMask,
                                                   oldFullMask, "not_value0&mask==mask");
            equalsMask = ctx->I1VecToBoolVec(equalsMask);
            if (!ctx->emitXeHardwareMask()) {
                llvm::Value *allMatch = ctx->All(equalsMask);
                ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, allMatch);
            } else {
                // If uniform CF is emulated, pass vector value to BranchInst
                ctx->BranchInst(bbSkipEvalValue1, bbEvalValue1, equalsMask);
            }

            // value0 was false for all running lanes, so use its value as
            // the overall result.
            ctx->SetCurrentBasicBlock(bbSkipEvalValue1);
            ctx->StoreInst(value0, retPtrInfo, arg0->GetType());
            ctx->BranchInst(bbLogicalDone);

            // Otherwise we need to evaluate value1, but again with the
            // mask set to only be on for the lanes where value0 was true.
            // For the lanes where value0 was false, execution needs to be
            // disabled: mask = (mask & value0).
            ctx->SetCurrentBasicBlock(bbEvalValue1);
            ctx->SetInternalMaskAnd(oldMask, value0);

            llvm::Value *value1 = arg1->GetValue(ctx);
            if (value1 == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                return nullptr;
            }

            // And as in the || case, we compute the overall result by
            // masking off the valid lanes before we AND them together:
            // result = (value0 & old_mask) & (value1 & current_mask)
            llvm::Value *value0AndMask =
                ctx->BinaryOperator(llvm::Instruction::And, value0, oldFullMask, WrapSemantics::None, "op&mask");
            llvm::Value *value1AndMask = ctx->BinaryOperator(llvm::Instruction::And, value1, ctx->GetInternalMask(),
                                                             WrapSemantics::None, "value1&mask");
            llvm::Value *result = ctx->BinaryOperator(llvm::Instruction::And, value0AndMask, value1AndMask,
                                                      WrapSemantics::None, "or_result");
            ctx->StoreInst(result, retPtrInfo, retType);
            ctx->BranchInst(bbLogicalDone);
        }

        // And finally we always end up in bbLogicalDone, where we restore
        // the old mask and return the computed result
        ctx->SetCurrentBasicBlock(bbLogicalDone);
        ctx->SetInternalMask(oldMask);
        return ctx->LoadInst(retPtrInfo, retType);
    }
}

/* Returns true if shifting right by the given amount will lead to
   inefficient code.  (Assumes x86 target.  May also warn inaccurately if
   later optimization simplify the shift amount more than we are able to
   see at this point.) */
static bool lIsDifficultShiftAmount(Expr *expr) {
    // Uniform shifts (of uniform values) are no problem.
    const Type *exprType = expr->GetType();
    Assert(exprType);
    if (exprType->IsVaryingType() == false)
        return false;

    ConstExpr *ce = llvm::dyn_cast<ConstExpr>(expr);
    if (ce) {
        // If the shift is by a constant amount, *and* it's the same amount
        // in all vector lanes, we're in good shape.
        uint32_t amount[ISPC_MAX_NVEC];
        int count = ce->GetValues(amount);
        for (int i = 1; i < count; ++i)
            if (amount[i] != amount[0])
                return true;
        return false;
    }

    TypeCastExpr *tce = llvm::dyn_cast<TypeCastExpr>(expr);
    if (tce && tce->expr) {
        // Finally, if the shift amount is given by a uniform value that's
        // been smeared out into a varying, we have the same shift for all
        // lanes and are also in good shape.
        const Type *tceType = tce->expr->GetType();
        Assert(tceType);
        return (tceType->IsUniformType() == false);
    }

    return true;
}

bool BinaryExpr::HasAmbiguousVariability(std::vector<const Expr *> &warn) const {
    bool isArg0Amb = false;
    bool isArg1Amb = false;
    if (arg0 != nullptr) {
        const Type *type0 = arg0->GetType();
        if (arg0->HasAmbiguousVariability(warn)) {
            isArg0Amb = true;
        } else if ((type0 != nullptr) && (type0->IsVaryingType())) {
            // If either arg is varying, then the expression is un-ambiguously varying.
            return false;
        }
    }
    if (arg1 != nullptr) {
        const Type *type1 = arg1->GetType();
        if (arg1->HasAmbiguousVariability(warn)) {
            isArg1Amb = true;
        } else if ((type1 != nullptr) && (type1->IsVaryingType())) {
            // If either arg is varying, then the expression is un-ambiguously varying.
            return false;
        }
    }
    if (isArg0Amb || isArg1Amb) {
        return true;
    }

    return false;
}

llvm::Value *BinaryExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!arg0 || !arg1) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    // Handle these specially, since we want to short-circuit their evaluation...
    if (op == LogicalAnd || op == LogicalOr)
        return lEmitLogicalOp(op, arg0, arg1, ctx, pos);

    llvm::Value *value0 = arg0->GetValue(ctx);
    llvm::Value *value1 = arg1->GetValue(ctx);
    if (value0 == nullptr || value1 == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    ctx->SetDebugPos(pos);
    switch (op) {
    case Add:
    case Sub:
    case Mul: {
        const bool isSigned = this->GetType()->IsSignedType();
        return lEmitBinaryArith(op, value0, value1, arg0->GetType(), arg1->GetType(), ctx, pos,
                                (isSigned ? WrapSemantics::NSW : WrapSemantics::None));
    }
    case Div:
    case Mod:
        return lEmitBinaryArith(op, value0, value1, arg0->GetType(), arg1->GetType(), ctx, pos, WrapSemantics::None);
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
        if (op == Shr && lIsDifficultShiftAmount(arg1) && g->target->shouldWarn(PerfWarningType::VariableShiftRight)) {
            PerformanceWarning(pos, "Shift right is inefficient for varying shift amounts.");
        }
        return lEmitBinaryBitOp(op, value0, value1, this->GetType()->IsUnsignedType(), ctx);
    }
    case Comma:
        return value1;
    default:
        FATAL("logic error");
        return nullptr;
    }
}

const Type *BinaryExpr::GetType() const {
    if (arg0 == nullptr || arg1 == nullptr)
        return nullptr;

    const Type *type0 = arg0->GetType(), *type1 = arg1->GetType();
    if (type0 == nullptr || type1 == nullptr)
        return nullptr;

    if (type0->IsDependentType() || type1->IsDependentType()) {
        return AtomicType::Dependent;
    }

    // If this hits, it means that our TypeCheck() method hasn't been
    // called before GetType() was called; adding two pointers is illegal
    // and will fail type checking and (int + ptr) should be canonicalized
    // into (ptr + int) by type checking.
    if (op == Add)
        AssertPos(pos, CastType<PointerType>(type1) == nullptr);

    if (op == Comma)
        return arg1->GetType();

    if (CastType<PointerType>(type0) != nullptr) {
        if (op == Add)
            // ptr + int -> ptr
            return type0;
        else if (op == Sub) {
            if (CastType<PointerType>(type1) != nullptr) {
                // ptr - ptr -> ~ptrdiff_t
                const Type *diffType = (g->target->is32Bit() || g->opt.force32BitAddressing) ? AtomicType::UniformInt32
                                                                                             : AtomicType::UniformInt64;
                if (type0->IsVaryingType() || type1->IsVaryingType())
                    diffType = diffType->GetAsVaryingType();
                return diffType;
            } else
                // ptr - int -> ptr
                return type0;
        }

        // otherwise fall through for these...
        AssertPos(pos, op == Lt || op == Gt || op == Le || op == Ge || op == Equal || op == NotEqual);
    }

    const Type *exprType = Type::MoreGeneralType(type0, type1, pos, lOpString(op));
    // I don't think that MoreGeneralType should be able to fail after the
    // checks done in BinaryExpr::TypeCheck().
    AssertPos(pos, exprType != nullptr);

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
        return nullptr;
    }
}

#define FOLD_OP(O, E)                                                                                                  \
    case O:                                                                                                            \
        for (int i = 0; i < count; ++i)                                                                                \
            result[i] = (v0[i] E v1[i]);                                                                               \
        break

#define FOLD_FP_OP(O, E)                                                                                               \
    case O:                                                                                                            \
        for (int i = 0; i < count; ++i) {                                                                              \
            llvm::APFloat temp(v0[i]);                                                                                 \
            llvm::APFloatBase::opStatus status = temp.E(v1[i], llvm::APFloat::rmNearestTiesToEven);                    \
            lPrintConstFoldBinaryArithFPWarning(carg0, status, pos);                                                   \
            result.push_back(temp);                                                                                    \
        }                                                                                                              \
        break

#define FOLD_OP_REF(O, E, TRef)                                                                                        \
    case O:                                                                                                            \
        for (int i = 0; i < count; ++i) {                                                                              \
            result[i] = (v0[i] E v1[i]);                                                                               \
            TRef r = (TRef)v0[i] E(TRef) v1[i];                                                                        \
            if (result[i] != r)                                                                                        \
                Warning(pos, "Binary expression with type \"%s\" can't represent value.",                              \
                        carg0->GetType()->GetString().c_str());                                                        \
        }                                                                                                              \
        break

template <typename T> static int countLeadingZeros(T val) {

    int leadingZeros = 0;
    size_t size = sizeof(T) * CHAR_BIT;
    T msb = (T)(T(1) << (size - 1));

    while (size--) {
        if (msb & val) {
            break;
        }
        msb = msb >> 1;
        leadingZeros++;
    }
    return leadingZeros;
}

/** Constant fold the binary integer operations that aren't also applicable
    to floating-point types.
*/
template <typename T, typename TRef>
static ConstExpr *lConstFoldBinaryIntOp(BinaryExpr::Op op, const T *v0, const T *v1, ConstExpr *carg0, SourcePos pos) {
    T result[ISPC_MAX_NVEC];
    int count = carg0->Count();

    switch (op) {
        FOLD_OP_REF(BinaryExpr::Shr, >>, TRef);
        FOLD_OP_REF(BinaryExpr::BitAnd, &, TRef);
        FOLD_OP_REF(BinaryExpr::BitXor, ^, TRef);
        FOLD_OP_REF(BinaryExpr::BitOr, |, TRef);

    case BinaryExpr::Shl:
        for (int i = 0; i < count; ++i) {
            result[i] = (T(v0[i]) << v1[i]);
            if (v1[i] > countLeadingZeros(v0[i])) {
                Warning(pos, "Binary expression with type \"%s\" can't represent value.",
                        carg0->GetType()->GetString().c_str());
            }
        }
        break;
    case BinaryExpr::Mod:
        for (int i = 0; i < count; ++i) {
            if (v1[i] == 0) {
                Warning(pos, "Remainder by zero is undefined.");
                return nullptr;
            } else {
                result[i] = (v0[i] % v1[i]);
            }
        }
        break;
    default:
        return nullptr;
    }

    return new ConstExpr(carg0->GetType(), result, carg0->pos);
}

/** Constant fold the binary logical ops.
 */
template <typename T>
static ConstExpr *lConstFoldBinaryLogicalOp(BinaryExpr::Op op, const T *v0, const T *v1, ConstExpr *carg0) {
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
        return nullptr;
    }

    const Type *rType = carg0->GetType()->IsUniformType() ? AtomicType::UniformBool : AtomicType::VaryingBool;
    return new ConstExpr(rType, result, carg0->pos);
}

static ConstExpr *lConstFoldBinaryLogicalFPOp(BinaryExpr::Op op, std::vector<llvm::APFloat> const &v0,
                                              std::vector<llvm::APFloat> const &v1, ConstExpr *carg0) {
    bool result[ISPC_MAX_NVEC];
    int count = carg0->Count();

    switch (op) {
        FOLD_OP(BinaryExpr::Lt, <);
        FOLD_OP(BinaryExpr::Gt, >);
        FOLD_OP(BinaryExpr::Le, <=);
        FOLD_OP(BinaryExpr::Ge, >=);
        FOLD_OP(BinaryExpr::Equal, ==);
        FOLD_OP(BinaryExpr::NotEqual, !=);
    case BinaryExpr::LogicalAnd:
        for (int i = 0; i < count; ++i) {
            result[i] = v0[i].isNonZero() && v1[i].isNonZero();
        }
        break;
    case BinaryExpr::LogicalOr:
        for (int i = 0; i < count; ++i) {
            result[i] = v0[i].isNonZero() || v1[i].isNonZero();
        }
        break;
    default:
        return nullptr;
    }

    const Type *rType = carg0->GetType()->IsUniformType() ? AtomicType::UniformBool : AtomicType::VaryingBool;
    return new ConstExpr(rType, result, carg0->pos);
}

/** Constant fold binary arithmetic ops for Floating Point values.
 */
static void lPrintConstFoldBinaryArithFPWarning(ConstExpr *carg0, llvm::APFloatBase::opStatus status, SourcePos pos) {
    switch (status) {
    case llvm::APFloatBase::opStatus::opInvalidOp:
    case llvm::APFloatBase::opStatus::opOverflow:
    case llvm::APFloatBase::opStatus::opUnderflow:
        Warning(pos, "Binary expression with type \"%s\" can't represent value.",
                carg0->GetType()->GetString().c_str());
        break;
    case llvm::APFloatBase::opStatus::opDivByZero:
        Warning(pos, "Division by zero is undefined.");
    default:
        break;
    }
}

static ConstExpr *lConstFoldBinaryArithFPOp(BinaryExpr::Op op, std::vector<llvm::APFloat> const &v0,
                                            std::vector<llvm::APFloat> const &v1, ConstExpr *carg0, SourcePos pos) {
    std::vector<llvm::APFloat> result;
    int count = carg0->Count();

    switch (op) {
        FOLD_FP_OP(BinaryExpr::Add, add);
        FOLD_FP_OP(BinaryExpr::Sub, subtract);
        FOLD_FP_OP(BinaryExpr::Mul, multiply);
        FOLD_FP_OP(BinaryExpr::Div, divide);
    default:
        return nullptr;
    }

    return new ConstExpr(carg0->GetType(), result, carg0->pos);
}

/** Constant fold binary arithmetic ops.
 */
template <typename T, typename TRef>
static ConstExpr *lConstFoldBinaryArithOp(BinaryExpr::Op op, const T *v0, const T *v1, ConstExpr *carg0,
                                          SourcePos pos) {
    T result[ISPC_MAX_NVEC];
    int count = carg0->Count();

    switch (op) {
        FOLD_OP_REF(BinaryExpr::Add, +, TRef);
        FOLD_OP_REF(BinaryExpr::Sub, -, TRef);
        FOLD_OP_REF(BinaryExpr::Mul, *, TRef);
    case BinaryExpr::Div:
        for (int i = 0; i < count; ++i) {
            if (v1[i] == 0) {
                Warning(pos, "Division by zero is undefined.");
                return nullptr;
            } else {
                result[i] = (v0[i] / v1[i]);
            }
        }
        break;
    default:
        return nullptr;
    }

    return new ConstExpr(carg0->GetType(), result, carg0->pos);
}

/** Constant fold the various boolean binary ops.
 */
static ConstExpr *lConstFoldBoolBinaryOp(BinaryExpr::Op op, const bool *v0, const bool *v1, ConstExpr *carg0) {
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
        return nullptr;
    }

    return new ConstExpr(carg0->GetType(), result, carg0->pos);
}

static Expr *lConstFoldBinaryFPOp(ConstExpr *constArg0, ConstExpr *constArg1, BinaryExpr::Op op, BinaryExpr *origExpr,
                                  llvm::Type *llvmType, SourcePos pos) {
    std::vector<llvm::APFloat> v0, v1;
    constArg0->GetValues(v0, llvmType);
    constArg1->GetValues(v1, llvmType);
    ConstExpr *ret;
    if ((ret = lConstFoldBinaryArithFPOp(op, v0, v1, constArg0, pos)) != nullptr)
        return ret;
    else if ((ret = lConstFoldBinaryLogicalFPOp(op, v0, v1, constArg0)) != nullptr)
        return ret;
    else
        return origExpr;
}

template <typename T, typename TRef>
static Expr *lConstFoldBinaryIntOp(ConstExpr *constArg0, ConstExpr *constArg1, BinaryExpr::Op op, BinaryExpr *origExpr,
                                   SourcePos pos) {
    T v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
    constArg0->GetValues(v0);
    constArg1->GetValues(v1);
    ConstExpr *ret;
    if ((ret = lConstFoldBinaryArithOp<T, TRef>(op, v0, v1, constArg0, pos)) != nullptr)
        return ret;
    else if ((ret = lConstFoldBinaryIntOp<T, TRef>(op, v0, v1, constArg0, pos)) != nullptr)
        return ret;
    else if ((ret = lConstFoldBinaryLogicalOp(op, v0, v1, constArg0)) != nullptr)
        return ret;
    else
        return origExpr;
}

Expr *BinaryExpr::Optimize() {
    if (arg0 == nullptr || arg1 == nullptr)
        return nullptr;

    ConstExpr *constArg0 = llvm::dyn_cast<ConstExpr>(arg0);
    ConstExpr *constArg1 = llvm::dyn_cast<ConstExpr>(arg1);

    if (g->opt.fastMath) {
        // TODO: consider moving fast-math optimizations to backend

        // optimizations related to division by floats..

        // transform x / const -> x * (1/const)
        if (op == Div && constArg1 != nullptr) {
            const Type *type1 = constArg1->GetType();
            if (Type::EqualIgnoringConst(type1, AtomicType::UniformFloat16) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingFloat16) ||
                Type::EqualIgnoringConst(type1, AtomicType::UniformFloat) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingFloat) ||
                Type::EqualIgnoringConst(type1, AtomicType::UniformDouble) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingDouble)) {
                llvm::Type *eltType = type1->GetAsUniformType()->LLVMType(g->ctx);
                std::vector<llvm::APFloat> constVal;
                int count = constArg1->GetValues(constVal, eltType);
                std::vector<llvm::APFloat> inv;
                for (int i = 0; i < count; ++i) {
                    llvm::APFloat reciprocal = lCreateAPFloat(1.0, eltType);
                    reciprocal.divide(constVal[i], llvm::APFloat::rmNearestTiesToEven);
                    inv.push_back(reciprocal);
                }
                Expr *einv = new ConstExpr(type1, inv, constArg1->pos);
                Expr *e = new BinaryExpr(Mul, arg0, einv, pos);
                e = ::TypeCheck(e);
                if (e == nullptr)
                    return nullptr;
                return ::Optimize(e);
            }
        }

        // transform x / y -> x * rcp(y)
        if (op == Div) {
            const Type *type1 = arg1->GetType();
            if (Type::EqualIgnoringConst(type1, AtomicType::UniformFloat16) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingFloat16) ||
                Type::EqualIgnoringConst(type1, AtomicType::UniformFloat) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingFloat) ||
                Type::EqualIgnoringConst(type1, AtomicType::UniformDouble) ||
                Type::EqualIgnoringConst(type1, AtomicType::VaryingDouble)) {
                // Get the symbol for the appropriate builtin
                std::vector<Symbol *> rcpFuns;
                m->symbolTable->LookupFunction("rcp", &rcpFuns);
                if (rcpFuns.size() > 0) {
                    Expr *rcpSymExpr = new FunctionSymbolExpr("rcp", rcpFuns, pos);
                    ExprList *args = new ExprList(arg1, arg1->pos);
                    Expr *rcpCall = new FunctionCallExpr(rcpSymExpr, args, arg1->pos);
                    rcpCall = ::TypeCheck(rcpCall);
                    if (rcpCall != nullptr) {
                        rcpCall = ::Optimize(rcpCall);
                        if (rcpCall != nullptr) {
                            Expr *ret = new BinaryExpr(Mul, arg0, rcpCall, pos);
                            ret = ::TypeCheck(ret);
                            if (ret == nullptr)
                                return nullptr;
                            return ::Optimize(ret);
                        }
                    }
                }

                Warning(pos,
                        "rcp(%s) not found from stdlib.  Can't apply "
                        "fast-math rcp optimization.",
                        type1->GetString().c_str());
            }
        }
    }

    // From here on out, we're just doing constant folding, so if both args
    // aren't constants then we're done...
    if (constArg0 == nullptr || constArg1 == nullptr)
        return this;

    AssertPos(pos, Type::EqualIgnoringConst(arg0->GetType(), arg1->GetType()));
    const Type *type = arg0->GetType()->GetAsNonConstType();
    if (Type::Equal(type, AtomicType::UniformFloat16) || Type::Equal(type, AtomicType::VaryingFloat16)) {
        return lConstFoldBinaryFPOp(constArg0, constArg1, op, this, LLVMTypes::Float16Type, pos);
    } else if (Type::Equal(type, AtomicType::UniformFloat) || Type::Equal(type, AtomicType::VaryingFloat)) {
        return lConstFoldBinaryFPOp(constArg0, constArg1, op, this, LLVMTypes::FloatType, pos);
    } else if (Type::Equal(type, AtomicType::UniformDouble) || Type::Equal(type, AtomicType::VaryingDouble)) {
        return lConstFoldBinaryFPOp(constArg0, constArg1, op, this, LLVMTypes::DoubleType, pos);
    } else if (Type::Equal(type, AtomicType::UniformInt8) || Type::Equal(type, AtomicType::VaryingInt8)) {
        return lConstFoldBinaryIntOp<int8_t, int64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformUInt8) || Type::Equal(type, AtomicType::VaryingUInt8)) {
        return lConstFoldBinaryIntOp<uint8_t, uint64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformInt16) || Type::Equal(type, AtomicType::VaryingInt16)) {
        return lConstFoldBinaryIntOp<int16_t, int64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformUInt16) || Type::Equal(type, AtomicType::VaryingUInt16)) {
        return lConstFoldBinaryIntOp<uint16_t, uint64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformInt32) || Type::Equal(type, AtomicType::VaryingInt32)) {
        return lConstFoldBinaryIntOp<int32_t, int64_t>(constArg0, constArg1, op, this, pos);
    } else if (CastType<EnumType>(type)) {
        return lConstFoldBinaryIntOp<uint32_t, uint64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformUInt32) || Type::Equal(type, AtomicType::VaryingUInt32)) {
        return lConstFoldBinaryIntOp<uint32_t, uint64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformInt64) || Type::Equal(type, AtomicType::VaryingInt64)) {
        return lConstFoldBinaryIntOp<int64_t, int64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformUInt64) || Type::Equal(type, AtomicType::VaryingUInt64)) {
        return lConstFoldBinaryIntOp<uint64_t, uint64_t>(constArg0, constArg1, op, this, pos);
    } else if (Type::Equal(type, AtomicType::UniformBool) || Type::Equal(type, AtomicType::VaryingBool)) {
        bool v0[ISPC_MAX_NVEC], v1[ISPC_MAX_NVEC];
        constArg0->GetValues(v0);
        constArg1->GetValues(v1);
        ConstExpr *ret;
        if ((ret = lConstFoldBoolBinaryOp(op, v0, v1, constArg0)) != nullptr)
            return ret;
        else if ((ret = lConstFoldBinaryLogicalOp(op, v0, v1, constArg0)) != nullptr)
            return ret;
        else
            return this;
    } else
        return this;
}

Expr *BinaryExpr::TypeCheck() {
    if (arg0 == nullptr || arg1 == nullptr)
        return nullptr;

    const Type *type0 = arg0->GetType(), *type1 = arg1->GetType();
    if (type0 == nullptr || type1 == nullptr)
        return nullptr;

    if (type0->IsDependentType() || type1->IsDependentType()) {
        return this;
    }

    // If either operand is a reference, dereference it before we move
    // forward
    if (CastType<ReferenceType>(type0) != nullptr) {
        arg0 = new RefDerefExpr(arg0, arg0->pos);
        type0 = arg0->GetType();
        AssertPos(pos, type0 != nullptr);
    }
    if (CastType<ReferenceType>(type1) != nullptr) {
        arg1 = new RefDerefExpr(arg1, arg1->pos);
        type1 = arg1->GetType();
        AssertPos(pos, type1 != nullptr);
    }

    // Convert arrays to pointers to their first elements
    if (CastType<ArrayType>(type0) != nullptr) {
        arg0 = lArrayToPointer(arg0);
        type0 = arg0->GetType();
    }
    if (CastType<ArrayType>(type1) != nullptr) {
        arg1 = lArrayToPointer(arg1);
        type1 = arg1->GetType();
    }

    // Prohibit binary operators with SOA types
    if (type0->GetSOAWidth() > 0) {
        Error(arg0->pos,
              "Illegal to use binary operator %s with SOA type "
              "\"%s\".",
              lOpString(op), type0->GetString().c_str());
        return nullptr;
    }
    if (type1->GetSOAWidth() > 0) {
        Error(arg1->pos,
              "Illegal to use binary operator %s with SOA type "
              "\"%s\".",
              lOpString(op), type1->GetString().c_str());
        return nullptr;
    }

    const PointerType *pt0 = CastType<PointerType>(type0);
    const PointerType *pt1 = CastType<PointerType>(type1);
    if (pt0 != nullptr && pt1 != nullptr && op == Sub) {
        // Pointer subtraction
        if (PointerType::IsVoidPointer(type0)) {
            Error(pos,
                  "Illegal to perform pointer arithmetic "
                  "on \"%s\" type.",
                  type0->GetString().c_str());
            return nullptr;
        }
        if (PointerType::IsVoidPointer(type1)) {
            Error(pos,
                  "Illegal to perform pointer arithmetic "
                  "on \"%s\" type.",
                  type1->GetString().c_str());
            return nullptr;
        }
        if (CastType<UndefinedStructType>(pt0->GetBaseType())) {
            Error(pos,
                  "Illegal to perform pointer arithmetic "
                  "on undefined struct type \"%s\".",
                  pt0->GetString().c_str());
            return nullptr;
        }
        if (CastType<UndefinedStructType>(pt1->GetBaseType())) {
            Error(pos,
                  "Illegal to perform pointer arithmetic "
                  "on undefined struct type \"%s\".",
                  pt1->GetString().c_str());
            return nullptr;
        }

        const Type *t = Type::MoreGeneralType(type0, type1, pos, "-");
        if (t == nullptr)
            return nullptr;

        arg0 = TypeConvertExpr(arg0, t, "pointer subtraction");
        arg1 = TypeConvertExpr(arg1, t, "pointer subtraction");
        if (arg0 == nullptr || arg1 == nullptr)
            return nullptr;

        return this;
    } else if (((pt0 != nullptr || pt1 != nullptr) && op == Add) || (pt0 != nullptr && op == Sub)) {
        // Handle ptr + int, int + ptr, ptr - int
        if (pt0 != nullptr && pt1 != nullptr) {
            Error(pos, "Illegal to add two pointer types \"%s\" and \"%s\".", pt0->GetString().c_str(),
                  pt1->GetString().c_str());
            return nullptr;
        } else if (pt1 != nullptr) {
            // put in canonical order with the pointer as the first operand
            // for GetValue()
            std::swap(arg0, arg1);
            std::swap(type0, type1);
            std::swap(pt0, pt1);
        }

        AssertPos(pos, pt0 != nullptr);

        if (PointerType::IsVoidPointer(pt0)) {
            Error(pos,
                  "Illegal to perform pointer arithmetic "
                  "on \"%s\" type.",
                  pt0->GetString().c_str());
            return nullptr;
        }
        if (CastType<UndefinedStructType>(pt0->GetBaseType())) {
            Error(pos,
                  "Illegal to perform pointer arithmetic "
                  "on undefined struct type \"%s\".",
                  pt0->GetString().c_str());
            return nullptr;
        }

        const Type *offsetType = g->target->is32Bit() ? AtomicType::UniformInt32 : AtomicType::UniformInt64;
        if (pt0->IsVaryingType())
            offsetType = offsetType->GetAsVaryingType();
        if (type1->IsVaryingType()) {
            arg0 = TypeConvertExpr(arg0, type0->GetAsVaryingType(), "pointer addition");
            offsetType = offsetType->GetAsVaryingType();
            AssertPos(pos, arg0 != nullptr);
        }

        arg1 = TypeConvertExpr(arg1, offsetType, lOpString(op));
        if (arg1 == nullptr)
            return nullptr;

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
            Error(arg0->pos,
                  "First operand to binary operator \"%s\" must be "
                  "an integer or bool.",
                  lOpString(op));
            return nullptr;
        }
        if (!type1->IsIntType() && !type1->IsBoolType()) {
            Error(arg1->pos,
                  "Second operand to binary operator \"%s\" must be "
                  "an integer or bool.",
                  lOpString(op));
            return nullptr;
        }

        if (op == Shl || op == Shr) {
            bool isVarying = (type0->IsVaryingType() || type1->IsVaryingType());
            if (isVarying) {
                arg0 = TypeConvertExpr(arg0, type0->GetAsVaryingType(), "shift operator");
                if (arg0 == nullptr)
                    return nullptr;
                type0 = arg0->GetType();
            }
            arg1 = TypeConvertExpr(arg1, type0, "shift operator");
            if (arg1 == nullptr)
                return nullptr;
        } else {
            const Type *promotedType = Type::MoreGeneralType(type0, type1, arg0->pos, "binary bit op");
            if (promotedType == nullptr)
                return nullptr;

            arg0 = TypeConvertExpr(arg0, promotedType, "binary bit op");
            arg1 = TypeConvertExpr(arg1, promotedType, "binary bit op");
            if (arg0 == nullptr || arg1 == nullptr)
                return nullptr;
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
            Error(arg0->pos,
                  "First operand to binary operator \"%s\" is of "
                  "invalid type \"%s\".",
                  lOpString(op), type0->GetString().c_str());
            return nullptr;
        }
        if (!type1->IsNumericType() || (op == Mod && type1->IsFloatType())) {
            Error(arg1->pos,
                  "First operand to binary operator \"%s\" is of "
                  "invalid type \"%s\".",
                  lOpString(op), type1->GetString().c_str());
            return nullptr;
        }

        const Type *promotedType = Type::MoreGeneralType(type0, type1, Union(arg0->pos, arg1->pos), lOpString(op));
        if (promotedType == nullptr)
            return nullptr;

        arg0 = TypeConvertExpr(arg0, promotedType, lOpString(op));
        arg1 = TypeConvertExpr(arg1, promotedType, lOpString(op));
        if (arg0 == nullptr || arg1 == nullptr)
            return nullptr;
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
        // pointer type to a nullptr pointer.
        if (pt0 != nullptr && lIsAllIntZeros(arg1)) {
            arg1 = new NullPointerExpr(pos);
            type1 = arg1->GetType();
            pt1 = CastType<PointerType>(type1);
        } else if (pt1 != nullptr && lIsAllIntZeros(arg0)) {
            arg0 = new NullPointerExpr(pos);
            type0 = arg0->GetType();
            pt0 = CastType<PointerType>(type0);
        }

        if (pt0 == nullptr && pt1 == nullptr) {
            if (!type0->IsBoolType() && !type0->IsNumericType()) {
                Error(arg0->pos,
                      "First operand to operator \"%s\" is of "
                      "non-comparable type \"%s\".",
                      lOpString(op), type0->GetString().c_str());
                return nullptr;
            }
            if (!type1->IsBoolType() && !type1->IsNumericType()) {
                Error(arg1->pos,
                      "Second operand to operator \"%s\" is of "
                      "non-comparable type \"%s\".",
                      lOpString(op), type1->GetString().c_str());
                return nullptr;
            }
        }

        const Type *promotedType = Type::MoreGeneralType(type0, type1, arg0->pos, lOpString(op));
        if (promotedType == nullptr)
            return nullptr;

        arg0 = TypeConvertExpr(arg0, promotedType, lOpString(op));
        arg1 = TypeConvertExpr(arg1, promotedType, lOpString(op));
        if (arg0 == nullptr || arg1 == nullptr)
            return nullptr;
        return this;
    }
    case LogicalAnd:
    case LogicalOr: {
        // For now, we just type convert to boolean types, of the same
        // variability as the original types.  (When generating code, it's
        // useful to have preserved the uniform/varying distinction.)
        const AtomicType *boolType0 = type0->IsUniformType() ? AtomicType::UniformBool : AtomicType::VaryingBool;
        const AtomicType *boolType1 = type1->IsUniformType() ? AtomicType::UniformBool : AtomicType::VaryingBool;

        const Type *destType0 = nullptr, *destType1 = nullptr;
        const VectorType *vtype0 = CastType<VectorType>(type0);
        const VectorType *vtype1 = CastType<VectorType>(type1);
        if (vtype0 && vtype1) {
            int sz0 = vtype0->GetElementCount(), sz1 = vtype1->GetElementCount();
            if (sz0 != sz1) {
                Error(pos,
                      "Can't do logical operation \"%s\" between vector types of "
                      "different sizes (%d vs. %d).",
                      lOpString(op), sz0, sz1);
                return nullptr;
            }
            destType0 = new VectorType(boolType0, sz0);
            destType1 = new VectorType(boolType1, sz1);
        } else if (vtype0 != nullptr) {
            destType0 = new VectorType(boolType0, vtype0->GetElementCount());
            destType1 = new VectorType(boolType1, vtype0->GetElementCount());
        } else if (vtype1 != nullptr) {
            destType0 = new VectorType(boolType0, vtype1->GetElementCount());
            destType1 = new VectorType(boolType1, vtype1->GetElementCount());
        } else {
            destType0 = boolType0;
            destType1 = boolType1;
        }

        arg0 = TypeConvertExpr(arg0, destType0, lOpString(op));
        arg1 = TypeConvertExpr(arg1, destType1, lOpString(op));
        if (arg0 == nullptr || arg1 == nullptr)
            return nullptr;
        return this;
    }
    case Comma:
        return this;
    default:
        FATAL("logic error");
        return nullptr;
    }
}

const Type *BinaryExpr::GetLValueType() const {
    const Type *t = GetType();
    if (CastType<PointerType>(t) != nullptr) {
        // Are we doing something like (basePtr + offset)[...] = ...
        return t;
    } else {
        return nullptr;
    }
}

int BinaryExpr::EstimateCost() const {
    if (llvm::dyn_cast<ConstExpr>(arg0) != nullptr && llvm::dyn_cast<ConstExpr>(arg1) != nullptr)
        return 0;

    return (op == Div || op == Mod) ? COST_COMPLEX_ARITH_OP : COST_SIMPLE_ARITH_LOGIC_OP;
}

Expr *BinaryExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instArg0 = arg0 ? arg0->Instantiate(templInst) : nullptr;
    Expr *instArg1 = arg1 ? arg1->Instantiate(templInst) : nullptr;
    return MakeBinaryExpr(op, instArg0, instArg1, pos);
}

void BinaryExpr::Print(Indent &indent) const {
    if (!arg0 || !arg1 || !GetType()) {
        indent.Print("BinaryExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("BinaryExpr", pos);

    printf("[ %s ], '%s'\n", GetType()->GetString().c_str(), lOpString(op));
    indent.pushList(2);
    arg0->Print(indent);
    arg1->Print(indent);

    indent.Done();
}

static std::pair<llvm::Constant *, bool> lGetBinaryExprStorageConstant(const Type *type, const BinaryExpr *bExpr,
                                                                       bool isStorageType) {

    const BinaryExpr::Op op = bExpr->op;
    Expr *arg0 = bExpr->arg0;
    Expr *arg1 = bExpr->arg1;

    // Are we doing something like (basePtr + offset)[...] = ... for a Global
    // Variable
    if (!bExpr->GetLValueType())
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    // We are limiting cases to just addition and subtraction involving
    // pointer addresses
    // Case 1 : first argument is a pointer address.
    // In this case as long as the second argument is a constant value, we are fine
    // Case 2 : second argument is a pointer address.
    // In this case, it has to be an addition with first argument as
    // a constant value.
    if (!((op == BinaryExpr::Op::Add) || (op == BinaryExpr::Op::Sub)))
        return std::pair<llvm::Constant *, bool>(nullptr, false);
    if (op == BinaryExpr::Op::Sub) {
        // Ignore cases where subtrahend is a PointerType
        // Eg. b - 5 is valid but 5 - b is not.
        if (CastType<PointerType>(arg1->GetType()))
            return std::pair<llvm::Constant *, bool>(nullptr, false);
    }

    // 'isNotValidForMultiTargetGlobal' is required to let the caller know
    // that the llvm::constant value returned cannot be used in case of
    // multi-target compilation for initialization of globals. This is due
    // to different constant values for different targets, i.e. computation
    // involving sizeof() of varying types.
    // Since converting expr to constant can be a recursive process, we need
    // to ensure that if the flag is set by any expr in the chain, it's
    // reflected in the final return value.
    bool isNotValidForMultiTargetGlobal = false;
    if (const PointerType *pt0 = CastType<PointerType>(arg0->GetType())) {
        std::pair<llvm::Constant *, bool> c1Pair;
        if (isStorageType)
            c1Pair = arg0->GetStorageConstant(pt0);
        else
            c1Pair = arg0->GetConstant(pt0);
        llvm::Constant *c1 = c1Pair.first;
        isNotValidForMultiTargetGlobal = isNotValidForMultiTargetGlobal || c1Pair.second;
        ConstExpr *cExpr = llvm::dyn_cast<ConstExpr>(arg1);
        if ((cExpr == nullptr) || (c1 == nullptr))
            return std::pair<llvm::Constant *, bool>(nullptr, false);
        std::pair<llvm::Constant *, bool> c2Pair;
        if (isStorageType)
            c2Pair = cExpr->GetStorageConstant(cExpr->GetType());
        else
            c2Pair = cExpr->GetConstant(cExpr->GetType());
        llvm::Constant *c2 = c2Pair.first;
        isNotValidForMultiTargetGlobal = isNotValidForMultiTargetGlobal || c2Pair.second;
        if (op == BinaryExpr::Op::Sub)
            c2 = llvm::ConstantExpr::getNeg(c2);
        llvm::Constant *c = llvm::ConstantExpr::getGetElementPtr(AddressInfo::GetPointeeLLVMType(pt0), c1, c2);
        return std::pair<llvm::Constant *, bool>(c, isNotValidForMultiTargetGlobal);
    } else if (const PointerType *pt1 = CastType<PointerType>(arg1->GetType())) {
        std::pair<llvm::Constant *, bool> c1Pair;
        if (isStorageType)
            c1Pair = arg1->GetStorageConstant(pt1);
        else
            c1Pair = arg1->GetConstant(pt1);
        llvm::Constant *c1 = c1Pair.first;
        isNotValidForMultiTargetGlobal = isNotValidForMultiTargetGlobal || c1Pair.second;
        ConstExpr *cExpr = llvm::dyn_cast<ConstExpr>(arg0);
        if ((cExpr == nullptr) || (c1 == nullptr))
            return std::pair<llvm::Constant *, bool>(nullptr, false);
        std::pair<llvm::Constant *, bool> c2Pair;
        if (isStorageType)
            c2Pair = cExpr->GetStorageConstant(cExpr->GetType());
        else
            c2Pair = cExpr->GetConstant(cExpr->GetType());
        llvm::Constant *c2 = c2Pair.first;
        isNotValidForMultiTargetGlobal = isNotValidForMultiTargetGlobal || c2Pair.second;
        llvm::Constant *c = llvm::ConstantExpr::getGetElementPtr(AddressInfo::GetPointeeLLVMType(pt1), c1, c2);
        return std::pair<llvm::Constant *, bool>(c, isNotValidForMultiTargetGlobal);
    }

    return std::pair<llvm::Constant *, bool>(nullptr, false);
}

std::pair<llvm::Constant *, bool> BinaryExpr::GetStorageConstant(const Type *type) const {
    return lGetBinaryExprStorageConstant(type, this, true);
}

std::pair<llvm::Constant *, bool> BinaryExpr::GetConstant(const Type *type) const {

    return lGetBinaryExprStorageConstant(type, this, false);
}
///////////////////////////////////////////////////////////////////////////
// AssignExpr

static const char *lOpString(AssignExpr::Op op) {
    switch (op) {
    case AssignExpr::Assign:
        return "=";
    case AssignExpr::MulAssign:
        return "*=";
    case AssignExpr::DivAssign:
        return "/=";
    case AssignExpr::ModAssign:
        return "%=";
    case AssignExpr::AddAssign:
        return "+=";
    case AssignExpr::SubAssign:
        return "-=";
    case AssignExpr::ShlAssign:
        return "<<=";
    case AssignExpr::ShrAssign:
        return ">>=";
    case AssignExpr::AndAssign:
        return "&=";
    case AssignExpr::XorAssign:
        return "^=";
    case AssignExpr::OrAssign:
        return "|=";
    default:
        FATAL("Missing op in lOpstring");
        return "";
    }
}

/** Emit code to do an "assignment + operation" operator, e.g. "+=".
 */
static llvm::Value *lEmitOpAssign(AssignExpr::Op op, Expr *arg0, Expr *arg1, const Type *type, Symbol *baseSym,
                                  SourcePos pos, FunctionEmitContext *ctx, WrapSemantics wrapSemantics) {
    llvm::Value *lv = arg0->GetLValue(ctx);
    if (!lv) {
        // FIXME: I think this test is unnecessary and that this case
        // should be caught during typechecking
        Error(pos, "Can't assign to left-hand side of expression.");
        return nullptr;
    }
    const Type *lvalueType = arg0->GetLValueType();
    const Type *resultType = arg0->GetType();
    if (lvalueType == nullptr || resultType == nullptr)
        return nullptr;

    // Get the value on the right-hand side of the assignment+operation
    // operator and load the current value on the left-hand side.
    llvm::Value *rvalue = arg1->GetValue(ctx);
    if (rvalue == nullptr)
        return nullptr;
    llvm::Value *mask = lMaskForSymbol(baseSym, ctx);
    ctx->SetDebugPos(arg0->pos);
    llvm::Value *oldLHS = ctx->LoadInst(lv, mask, lvalueType);
    ctx->SetDebugPos(pos);

    // Map the operator to the corresponding BinaryExpr::Op operator
    BinaryExpr::Op basicop;
    switch (op) {
    case AssignExpr::MulAssign:
        basicop = BinaryExpr::Mul;
        break;
    case AssignExpr::DivAssign:
        basicop = BinaryExpr::Div;
        break;
    case AssignExpr::ModAssign:
        basicop = BinaryExpr::Mod;
        break;
    case AssignExpr::AddAssign:
        basicop = BinaryExpr::Add;
        break;
    case AssignExpr::SubAssign:
        basicop = BinaryExpr::Sub;
        break;
    case AssignExpr::ShlAssign:
        basicop = BinaryExpr::Shl;
        break;
    case AssignExpr::ShrAssign:
        basicop = BinaryExpr::Shr;
        break;
    case AssignExpr::AndAssign:
        basicop = BinaryExpr::BitAnd;
        break;
    case AssignExpr::XorAssign:
        basicop = BinaryExpr::BitXor;
        break;
    case AssignExpr::OrAssign:
        basicop = BinaryExpr::BitOr;
        break;
    default:
        FATAL("logic error in lEmitOpAssign()");
        return nullptr;
    }

    // Emit the code to compute the new value
    llvm::Value *newValue = nullptr;
    switch (op) {
    case AssignExpr::DivAssign:
    case AssignExpr::ModAssign:
        newValue = lEmitBinaryArith(basicop, oldLHS, rvalue, type, arg1->GetType(), ctx, pos, WrapSemantics::None);
        break;
    case AssignExpr::MulAssign:
    case AssignExpr::AddAssign:
    case AssignExpr::SubAssign:
        newValue = lEmitBinaryArith(basicop, oldLHS, rvalue, type, arg1->GetType(), ctx, pos, wrapSemantics);
        break;
    case AssignExpr::ShlAssign:
    case AssignExpr::ShrAssign:
    case AssignExpr::AndAssign:
    case AssignExpr::XorAssign:
    case AssignExpr::OrAssign:
        newValue = lEmitBinaryBitOp(basicop, oldLHS, rvalue, arg0->GetType()->IsUnsignedType(), ctx);
        break;
    default:
        FATAL("logic error in lEmitOpAssign");
        return nullptr;
    }

    // And store the result back to the lvalue.
    ctx->SetDebugPos(arg0->pos);
    lStoreAssignResult(newValue, lv, resultType, lvalueType, ctx, baseSym);

    return newValue;
}

AssignExpr::AssignExpr(AssignExpr::Op o, Expr *a, Expr *b, SourcePos p) : Expr(p, AssignExprID), op(o) {
    lvalue = a;
    rvalue = b;
}

llvm::Value *AssignExpr::GetValue(FunctionEmitContext *ctx) const {
    const Type *type = nullptr;
    if (lvalue == nullptr || rvalue == nullptr || (type = GetType()) == nullptr)
        return nullptr;
    ctx->SetDebugPos(pos);

    Symbol *baseSym = lvalue->GetBaseSymbol();

    switch (op) {
    case Assign: {
        llvm::Value *ptr = lvalue->GetLValue(ctx);
        if (ptr == nullptr) {
            Error(lvalue->pos, "Left hand side of assignment expression can't "
                               "be assigned to.");
            return nullptr;
        }
        const Type *ptrType = lvalue->GetLValueType();
        const Type *valueType = rvalue->GetType();
        if (ptrType == nullptr || valueType == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }

        llvm::Value *value = rvalue->GetValue(ctx);
        if (value == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }

        ctx->SetDebugPos(lvalue->pos);

        lStoreAssignResult(value, ptr, valueType, ptrType, ctx, baseSym);

        return value;
    }
    case MulAssign:
    case AddAssign:
    case SubAssign: {
        // This should be caught during type checking
        AssertPos(pos, !CastType<ArrayType>(type) && !CastType<StructType>(type));
        const Type *lvalueType = lvalue->GetType();
        const Type *rvalueType = rvalue->GetType();
        if (lvalueType == nullptr || rvalueType == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }
        const auto wrapSemantics =
            (lvalueType->IsSignedType() && rvalueType->IsSignedType()) ? WrapSemantics::NSW : WrapSemantics::None;
        return lEmitOpAssign(op, lvalue, rvalue, type, baseSym, pos, ctx, wrapSemantics);
    }
    case DivAssign:
    case ModAssign:
    case ShlAssign:
    case ShrAssign:
    case AndAssign:
    case XorAssign:
    case OrAssign: {
        // This should be caught during type checking
        AssertPos(pos, !CastType<ArrayType>(type) && !CastType<StructType>(type));
        return lEmitOpAssign(op, lvalue, rvalue, type, baseSym, pos, ctx, WrapSemantics::None);
    }
    default:
        FATAL("logic error in AssignExpr::GetValue()");
        return nullptr;
    }
}

Expr *AssignExpr::Optimize() {
    if (lvalue == nullptr || rvalue == nullptr)
        return nullptr;
    return this;
}

const Type *AssignExpr::GetType() const {
    if (lvalue) {
        const Type *ltype = lvalue->GetType();
        if (ltype && ltype->IsDependentType()) {
            return AtomicType::Dependent;
        }
        return ltype;
    }
    return nullptr;
}

/** Recursively checks a structure type to see if it (or any struct type
    that it holds) has a const-qualified member. */
static bool lCheckForConstStructMember(SourcePos pos, const StructType *structType, const StructType *initialType) {
    for (int i = 0; i < structType->GetElementCount(); ++i) {
        const Type *t = structType->GetElementType(i);
        if (t->IsConstType()) {
            if (structType == initialType)
                Error(pos,
                      "Illegal to assign to type \"%s\" due to element "
                      "\"%s\" with type \"%s\".",
                      structType->GetString().c_str(), structType->GetElementName(i).c_str(), t->GetString().c_str());
            else
                Error(pos,
                      "Illegal to assign to type \"%s\" in type \"%s\" "
                      "due to element \"%s\" with type \"%s\".",
                      structType->GetString().c_str(), initialType->GetString().c_str(),
                      structType->GetElementName(i).c_str(), t->GetString().c_str());
            return true;
        }

        const StructType *st = CastType<StructType>(t);
        if (st != nullptr && lCheckForConstStructMember(pos, st, initialType))
            return true;
    }
    return false;
}

Expr *AssignExpr::TypeCheck() {
    if (lvalue == nullptr || rvalue == nullptr)
        return nullptr;

    const Type *ltype = lvalue->GetType();
    const Type *rtype = rvalue->GetType();
    if ((ltype && ltype->IsDependentType()) || (rtype && rtype->IsDependentType())) {
        return this;
    }

    bool lvalueIsReference = CastType<ReferenceType>(lvalue->GetType()) != nullptr;
    if (lvalueIsReference)
        lvalue = new RefDerefExpr(lvalue, lvalue->pos);

    if (PossiblyResolveFunctionOverloads(rvalue, lvalue->GetType()) == false) {
        Error(pos, "Unable to find overloaded function for function "
                   "pointer assignment.");
        return nullptr;
    }

    const Type *lhsType = lvalue->GetType();
    if (lhsType == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (lhsType->IsConstType()) {
        Error(lvalue->pos,
              "Can't assign to type \"%s\" on left-hand side of "
              "expression.",
              lhsType->GetString().c_str());
        return nullptr;
    }

    if (CastType<PointerType>(lhsType) != nullptr) {
        if (op == AddAssign || op == SubAssign) {
            if (PointerType::IsVoidPointer(lhsType)) {
                Error(pos,
                      "Illegal to perform pointer arithmetic on \"%s\" "
                      "type.",
                      lhsType->GetString().c_str());
                return nullptr;
            }

            const Type *deltaType = g->target->is32Bit() ? AtomicType::UniformInt32 : AtomicType::UniformInt64;
            if (lhsType->IsVaryingType())
                deltaType = deltaType->GetAsVaryingType();
            rvalue = TypeConvertExpr(rvalue, deltaType, lOpString(op));
        } else if (op == Assign)
            rvalue = TypeConvertExpr(rvalue, lhsType, "assignment");
        else {
            Error(lvalue->pos, "Assignment operator \"%s\" is illegal with pointer types.", lOpString(op));
            return nullptr;
        }
    } else if (CastType<ArrayType>(lhsType) != nullptr) {
        Error(lvalue->pos, "Illegal to assign to array type \"%s\".", lhsType->GetString().c_str());
        return nullptr;
    } else
        rvalue = TypeConvertExpr(rvalue, lhsType, lOpString(op));

    if (rvalue == nullptr)
        return nullptr;

    if (lhsType->IsFloatType() == true &&
        (op == ShlAssign || op == ShrAssign || op == AndAssign || op == XorAssign || op == OrAssign)) {
        Error(pos,
              "Illegal to use %s operator with floating-point "
              "operands.",
              lOpString(op));
        return nullptr;
    }

    const StructType *st = CastType<StructType>(lhsType);
    if (st != nullptr) {
        // Make sure we're not assigning to a struct that has a constant member
        if (lCheckForConstStructMember(pos, st, st))
            return nullptr;

        if (op != Assign) {
            Error(lvalue->pos,
                  "Assignment operator \"%s\" is illegal with struct "
                  "type \"%s\".",
                  lOpString(op), st->GetString().c_str());
            return nullptr;
        }
    }
    return this;
}

int AssignExpr::EstimateCost() const {
    if (op == Assign)
        return COST_ASSIGN;
    if (op == DivAssign || op == ModAssign)
        return COST_ASSIGN + COST_COMPLEX_ARITH_OP;
    else
        return COST_ASSIGN + COST_SIMPLE_ARITH_LOGIC_OP;
}

AssignExpr *AssignExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instLValue = lvalue ? lvalue->Instantiate(templInst) : nullptr;
    Expr *instRValue = rvalue ? rvalue->Instantiate(templInst) : nullptr;
    return new AssignExpr(op, instLValue, instRValue, pos);
}

void AssignExpr::Print(Indent &indent) const {
    if (!lvalue || !rvalue || !GetType()) {
        indent.Print("AssignExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("AssignExpr", pos);

    printf("[%s], '%s'\n", GetType()->GetString().c_str(), lOpString(op));
    indent.pushList(2);
    lvalue->Print(indent);
    rvalue->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// SelectExpr

SelectExpr::SelectExpr(Expr *t, Expr *e1, Expr *e2, SourcePos p) : Expr(p, SelectExprID) {
    test = t;
    expr1 = e1;
    expr2 = e2;
}

/** Emit code to select between two varying values based on a varying test
    value.
 */
static llvm::Value *lEmitVaryingSelect(FunctionEmitContext *ctx, llvm::Value *test, llvm::Value *expr1,
                                       llvm::Value *expr2, const Type *type) {

    AddressInfo *resultPtrInfo = ctx->AllocaInst(type, "selectexpr_tmp");
    Assert(resultPtrInfo != nullptr);
    // Don't need to worry about masking here
    ctx->StoreInst(expr2, resultPtrInfo, type);
    // Use masking to conditionally store the expr1 values
    Assert(resultPtrInfo->getType() == PointerType::GetUniform(type)->LLVMStorageType(g->ctx));
    ctx->StoreInst(expr1, resultPtrInfo->getPointer(), test, type, PointerType::GetUniform(type));
    return ctx->LoadInst(resultPtrInfo, type, "selectexpr_final");
}

static void lEmitSelectExprCode(FunctionEmitContext *ctx, llvm::Value *testVal, llvm::Value *oldMask,
                                llvm::Value *fullMask, Expr *expr, AddressInfo *exprPtrInfo) {
    llvm::BasicBlock *bbEval = ctx->CreateBasicBlock("select_eval_expr", ctx->GetCurrentBasicBlock());
    llvm::BasicBlock *bbDone = ctx->CreateBasicBlock("select_done", bbEval);

    // Check to see if the test was true for any of the currently executing
    // program instances.
    llvm::Value *testAndFullMask =
        ctx->BinaryOperator(llvm::Instruction::And, testVal, fullMask, WrapSemantics::None, "test&mask");
    llvm::Value *anyOn = ctx->Any(testAndFullMask);
    ctx->BranchInst(bbEval, bbDone, anyOn);

    ctx->SetCurrentBasicBlock(bbEval);
    llvm::Value *testAndMask =
        ctx->BinaryOperator(llvm::Instruction::And, testVal, oldMask, WrapSemantics::None, "test&mask");
    ctx->SetInternalMask(testAndMask);
    llvm::Value *exprVal = expr->GetValue(ctx);
    ctx->StoreInst(exprVal, exprPtrInfo, expr->GetType());
    ctx->BranchInst(bbDone);

    ctx->SetCurrentBasicBlock(bbDone);
}

bool SelectExpr::HasAmbiguousVariability(std::vector<const Expr *> &warn) const {
    bool isExpr1Amb = false;
    bool isExpr2Amb = false;
    if (expr1 != nullptr) {
        const Type *type1 = expr1->GetType();
        if (expr1->HasAmbiguousVariability(warn)) {
            isExpr1Amb = true;
        } else if ((type1 != nullptr) && (type1->IsVaryingType())) {
            // If either expr is varying, then the expression is un-ambiguously varying.
            return false;
        }
    }
    if (expr2 != nullptr) {
        const Type *type2 = expr2->GetType();
        if (expr2->HasAmbiguousVariability(warn)) {
            isExpr2Amb = true;
        } else if ((type2 != nullptr) && (type2->IsVaryingType())) {
            // If either arg is varying, then the expression is un-ambiguously varying.
            return false;
        }
    }
    if (isExpr1Amb || isExpr2Amb) {
        return true;
    }

    return false;
}

llvm::Value *SelectExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr1 || !expr2 || !test)
        return nullptr;

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
        llvm::BasicBlock *testTrue = ctx->CreateBasicBlock("select_true", ctx->GetCurrentBasicBlock());
        llvm::BasicBlock *testFalse = ctx->CreateBasicBlock("select_false", testTrue);
        llvm::BasicBlock *testDone = ctx->CreateBasicBlock("select_done", testFalse);
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
    } else if (CastType<VectorType>(testType) == nullptr) {
        // the test is a varying bool type
        llvm::Value *testVal = test->GetValue(ctx);
        AssertPos(pos, testVal->getType() == LLVMTypes::MaskType);
        llvm::Value *oldMask = ctx->GetInternalMask();
        llvm::Value *fullMask = ctx->GetFullMask();

        // We don't want to incur the overhead for short-circuit evaluation
        // for expressions that are both computationally simple and safe to
        // run with an "all off" mask.
        int threshold =
            g->target->isXeTarget() ? PREDICATE_SAFE_SHORT_CIRC_XE_STATEMENT_COST : PREDICATE_SAFE_IF_STATEMENT_COST;
        bool shortCircuit1 = (::EstimateCost(expr1) > threshold || SafeToRunWithMaskAllOff(expr1) == false);
        bool shortCircuit2 = (::EstimateCost(expr2) > threshold || SafeToRunWithMaskAllOff(expr2) == false);

        Debug(expr1->pos, "%sshort circuiting evaluation for select expr", shortCircuit1 ? "" : "Not ");
        Debug(expr2->pos, "%sshort circuiting evaluation for select expr", shortCircuit2 ? "" : "Not ");

        // Temporary storage to store the values computed for each
        // expression, if any.  (These stay as uninitialized memory if we
        // short circuit around the corresponding expression.)
        AddressInfo *expr1PtrInfo = ctx->AllocaInst(expr1->GetType());
        AddressInfo *expr2PtrInfo = ctx->AllocaInst(expr1->GetType());

        if (shortCircuit1)
            lEmitSelectExprCode(ctx, testVal, oldMask, fullMask, expr1, expr1PtrInfo);
        else {
            ctx->SetInternalMaskAnd(oldMask, testVal);
            llvm::Value *expr1Val = expr1->GetValue(ctx);
            ctx->StoreInst(expr1Val, expr1PtrInfo, expr1->GetType());
        }

        if (shortCircuit2) {
            llvm::Value *notTest = ctx->NotOperator(testVal);
            lEmitSelectExprCode(ctx, notTest, oldMask, fullMask, expr2, expr2PtrInfo);
        } else {
            ctx->SetInternalMaskAndNot(oldMask, testVal);
            llvm::Value *expr2Val = expr2->GetValue(ctx);
            ctx->StoreInst(expr2Val, expr2PtrInfo, expr2->GetType());
        }

        ctx->SetInternalMask(oldMask);
        llvm::Value *expr1Val = ctx->LoadInst(expr1PtrInfo, expr1->GetType());
        llvm::Value *expr2Val = ctx->LoadInst(expr2PtrInfo, expr2->GetType());
        return lEmitVaryingSelect(ctx, testVal, expr1Val, expr2Val, type);
    } else {
        // FIXME? Short-circuiting doesn't work in the case of
        // vector-valued test expressions.  (We could also just prohibit
        // these and place the issue in the user's hands...)
        llvm::Value *testVal = test->GetValue(ctx);
        llvm::Value *expr1Val = expr1->GetValue(ctx);
        llvm::Value *expr2Val = expr2->GetValue(ctx);

        ctx->SetDebugPos(pos);
        const VectorType *vt = CastType<VectorType>(type);
        // Things that typechecking should have caught
        AssertPos(pos, vt != nullptr);
        AssertPos(pos, CastType<VectorType>(testType) != nullptr &&
                           (CastType<VectorType>(testType)->GetElementCount() == vt->GetElementCount()));

        // Do an element-wise select
        llvm::Value *result = llvm::UndefValue::get(type->LLVMType(g->ctx));
        for (int i = 0; i < vt->GetElementCount(); ++i) {
            llvm::Value *ti = ctx->ExtractInst(testVal, i);
            llvm::Value *e1i = ctx->ExtractInst(expr1Val, i);
            llvm::Value *e2i = ctx->ExtractInst(expr2Val, i);
            llvm::Value *sel = nullptr;
            if (testType->IsUniformType()) {
                // Extracting uniform vector bool to uniform bool require
                // switching from i8 -> i1
                ti = ctx->SwitchBoolToMaskType(ti, LLVMTypes::BoolType);
                sel = ctx->SelectInst(ti, e1i, e2i);
            } else {
                // Extracting varying vector bools to varying bools require
                // switching from <WIDTH x i8> -> <WIDTH x MaskType>
                ti = ctx->SwitchBoolToMaskType(ti, LLVMTypes::BoolVectorType);
                sel = lEmitVaryingSelect(ctx, ti, e1i, e2i, vt->GetElementType());
            }
            result = ctx->InsertInst(result, sel, i);
        }
        return result;
    }
}

const Type *SelectExpr::GetType() const {
    if (!test || !expr1 || !expr2)
        return nullptr;

    const Type *testType = test->GetType();
    const Type *expr1Type = expr1->GetType();
    const Type *expr2Type = expr2->GetType();

    if (!testType || !expr1Type || !expr2Type)
        return nullptr;

    if (testType->IsDependentType() || expr1Type->IsDependentType() || expr2Type->IsDependentType()) {
        return AtomicType::Dependent;
    }

    bool becomesVarying = (testType->IsVaryingType() || expr1Type->IsVaryingType() || expr2Type->IsVaryingType());
    // if expr1 and expr2 have different vector sizes, typechecking should fail...
    int testVecSize = CastType<VectorType>(testType) != nullptr ? CastType<VectorType>(testType)->GetElementCount() : 0;
    int expr1VecSize =
        CastType<VectorType>(expr1Type) != nullptr ? CastType<VectorType>(expr1Type)->GetElementCount() : 0;
    AssertPos(pos, !(testVecSize != 0 && expr1VecSize != 0 && testVecSize != expr1VecSize));

    int vectorSize = std::max(testVecSize, expr1VecSize);
    return Type::MoreGeneralType(expr1Type, expr2Type, Union(expr1->pos, expr2->pos), "select expression",
                                 becomesVarying, vectorSize);
}

const Type *SelectExpr::GetLValueType() const {
    const Type *t = GetType();
    if (CastType<PointerType>(t) != nullptr) {
        return t;
    } else {
        return nullptr;
    }
}

template <typename T>
Expr *lConstFoldSelect(const bool bv[], ConstExpr *constExpr1, ConstExpr *constExpr2, const Type *exprType,
                       SourcePos pos) {
    T v1[ISPC_MAX_NVEC], v2[ISPC_MAX_NVEC];
    T result[ISPC_MAX_NVEC];
    int count = constExpr1->GetValues(v1);
    constExpr2->GetValues(v2);
    for (int i = 0; i < count; ++i)
        result[i] = bv[i] ? v1[i] : v2[i];
    return new ConstExpr(exprType, result, pos);
}

Expr *lConstFoldSelectFP(const bool bv[], ConstExpr *constExpr1, ConstExpr *constExpr2, const Type *exprType,
                         llvm::Type *llvmType, SourcePos pos) {
    std::vector<llvm::APFloat> v1, v2;
    std::vector<llvm::APFloat> result;
    int count = constExpr1->GetValues(v1, llvmType);
    constExpr2->GetValues(v2, llvmType);
    for (int i = 0; i < count; ++i)
        result.push_back(bv[i] ? v1[i] : v2[i]);
    return new ConstExpr(exprType, result, pos);
}

Expr *SelectExpr::Optimize() {
    if (test == nullptr || expr1 == nullptr || expr2 == nullptr)
        return nullptr;

    ConstExpr *constTest = llvm::dyn_cast<ConstExpr>(test);
    if (constTest == nullptr)
        return this;

    // The test is a constant; see if we can resolve to one of the
    // expressions..
    bool bv[ISPC_MAX_NVEC];
    int count = constTest->GetValues(bv);
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
        ConstExpr *constExpr1 = llvm::dyn_cast<ConstExpr>(expr1);
        ConstExpr *constExpr2 = llvm::dyn_cast<ConstExpr>(expr2);
        if (constExpr1 == nullptr || constExpr2 == nullptr)
            return this;

        AssertPos(pos, Type::Equal(constExpr1->GetType(), constExpr2->GetType()));
        const Type *exprType = constExpr1->GetType()->GetAsNonConstType();
        AssertPos(pos, exprType->IsVaryingType());

        if (Type::Equal(exprType, AtomicType::VaryingInt8)) {
            return lConstFoldSelect<int8_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingUInt8)) {
            return lConstFoldSelect<uint8_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingInt16)) {
            return lConstFoldSelect<int16_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingUInt16)) {
            return lConstFoldSelect<uint16_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingInt32)) {
            return lConstFoldSelect<int32_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingUInt32)) {
            return lConstFoldSelect<uint32_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingInt64)) {
            return lConstFoldSelect<int64_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingUInt64)) {
            return lConstFoldSelect<uint64_t>(bv, constExpr1, constExpr2, exprType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingFloat16)) {
            return lConstFoldSelectFP(bv, constExpr1, constExpr2, exprType, LLVMTypes::Float16Type, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingFloat)) {
            return lConstFoldSelectFP(bv, constExpr1, constExpr2, exprType, LLVMTypes::FloatType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingDouble)) {
            return lConstFoldSelectFP(bv, constExpr1, constExpr2, exprType, LLVMTypes::DoubleType, pos);
        } else if (Type::Equal(exprType, AtomicType::VaryingBool)) {
            return lConstFoldSelect<bool>(bv, constExpr1, constExpr2, exprType, pos);
        }

        return this;
    }
}

Expr *SelectExpr::TypeCheck() {
    if (test == nullptr || expr1 == nullptr || expr2 == nullptr)
        return nullptr;

    const Type *type1 = expr1->GetType(), *type2 = expr2->GetType(), *testType = test->GetType();
    if (!type1 || !type2 || !testType)
        return nullptr;

    if (testType->IsDependentType() || type1->IsDependentType() || type2->IsDependentType()) {
        return this;
    }

    if (const ArrayType *at1 = CastType<ArrayType>(type1)) {
        expr1 = TypeConvertExpr(expr1, PointerType::GetUniform(at1->GetBaseType()), "select");
        if (expr1 == nullptr)
            return nullptr;
        type1 = expr1->GetType();
    }
    if (const ArrayType *at2 = CastType<ArrayType>(type2)) {
        expr2 = TypeConvertExpr(expr2, PointerType::GetUniform(at2->GetBaseType()), "select");
        if (expr2 == nullptr)
            return nullptr;
        type2 = expr2->GetType();
    }

    test = TypeConvertExpr(test, lMatchingBoolType(testType), "select");
    if (test == nullptr)
        return nullptr;
    testType = test->GetType();

    int testVecSize = CastType<VectorType>(testType) ? CastType<VectorType>(testType)->GetElementCount() : 0;
    const Type *promotedType = Type::MoreGeneralType(type1, type2, Union(expr1->pos, expr2->pos), "select expression",
                                                     testType->IsVaryingType(), testVecSize);

    // If the promoted type is a ReferenceType, the expression type will be
    // the reference target type since SelectExpr is always a rvalue.
    if (CastType<ReferenceType>(promotedType) != nullptr)
        promotedType = promotedType->GetReferenceTarget();

    if (promotedType == nullptr)
        return nullptr;

    expr1 = TypeConvertExpr(expr1, promotedType, "select");
    expr2 = TypeConvertExpr(expr2, promotedType, "select");
    if (expr1 == nullptr || expr2 == nullptr)
        return nullptr;

    return this;
}

int SelectExpr::EstimateCost() const { return COST_SELECT; }

SelectExpr *SelectExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instTest = test ? test->Instantiate(templInst) : nullptr;
    Expr *instExpr1 = expr1 ? expr1->Instantiate(templInst) : nullptr;
    Expr *instExpr2 = expr2 ? expr2->Instantiate(templInst) : nullptr;
    return new SelectExpr(instTest, instExpr1, instExpr2, pos);
}

void SelectExpr::Print(Indent &indent) const {
    if (!test || !expr1 || !expr2 || !GetType()) {
        indent.Print("SelectExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("SelectExpr", pos);

    printf("[%s]\n", GetType()->GetString().c_str());
    indent.pushList(3);
    test->Print(indent);
    expr1->Print(indent);
    expr2->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// FunctionCallExpr

FunctionCallExpr::FunctionCallExpr(Expr *f, ExprList *a, SourcePos p, bool il, Expr *lce[3], bool iis)
    : Expr(p, FunctionCallExprID), isLaunch(il), isInvoke(iis) {
    func = f;
    args = a;
    std::vector<const Expr *> warn;
    if (a->HasAmbiguousVariability(warn) == true) {
        for (auto w : warn) {
            const TypeCastExpr *tExpr = llvm::dyn_cast<TypeCastExpr>(w);
            tExpr->PrintAmbiguousVariability();
        }
    }
    if (lce != nullptr) {
        launchCountExpr[0] = lce[0];
        launchCountExpr[1] = lce[1];
        launchCountExpr[2] = lce[2];
    } else
        launchCountExpr[0] = launchCountExpr[1] = launchCountExpr[2] = nullptr;
}

static const Type *lGetFunctionType(Expr *func) {
    if (func == nullptr)
        return nullptr;

    const Type *type = func->GetType();
    if (type == nullptr)
        return nullptr;

    if (type->IsDependentType())
        return type;

    const FunctionType *ftype = CastType<FunctionType>(type);
    if (ftype == nullptr) {
        // Not a regular function symbol--is it a function pointer?
        if (CastType<PointerType>(type) != nullptr)
            ftype = CastType<FunctionType>(type->GetBaseType());
    }
    return ftype;
}

llvm::Value *FunctionCallExpr::GetValue(FunctionEmitContext *ctx) const {
    if (func == nullptr || args == nullptr)
        return nullptr;

    ctx->SetDebugPos(pos);

    llvm::Value *callee = func->GetValue(ctx);

    if (callee == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    const Type *type = lGetFunctionType(func);
    if (type->IsDependentType()) {
        Error(pos, "Can't call function with dependent type.");
    }
    const FunctionType *ft = CastType<FunctionType>(type);
    AssertPos(pos, ft != nullptr);
    bool isVoidFunc = ft->GetReturnType()->IsVoidType();

    // Automatically convert function call args to references if needed.
    // FIXME: this should move to the TypeCheck() method... (but the
    // GetLValue call below needs a FunctionEmitContext, which is
    // problematic...)
    std::vector<Expr *> callargs = args->exprs;

    // Specifically, this can happen if there's an error earlier during
    // overload resolution.
    if ((int)callargs.size() > ft->GetNumParameters()) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    for (unsigned int i = 0; i < callargs.size(); ++i) {
        Expr *argExpr = callargs[i];
        if (argExpr == nullptr)
            continue;

        const Type *paramType = ft->GetParameterType(i);

        const Type *argLValueType = argExpr->GetLValueType();
        if (argLValueType != nullptr && CastType<PointerType>(argLValueType) != nullptr &&
            argLValueType->IsVaryingType() && CastType<ReferenceType>(paramType) != nullptr) {
            Error(argExpr->pos,
                  "Illegal to pass a \"varying\" lvalue to a "
                  "reference parameter of type \"%s\".",
                  paramType->GetString().c_str());
            return nullptr;
        }

        // Do whatever type conversion is needed
        argExpr = TypeConvertExpr(argExpr, paramType, "function call argument");
        if (argExpr == nullptr)
            return nullptr;
        callargs[i] = argExpr;
    }

    // Fill in any default argument values needed.
    // FIXME: should we do this during type checking?
    for (int i = callargs.size(); i < ft->GetNumParameters(); ++i) {
        Expr *paramDefault = ft->GetParameterDefault(i);
        const Type *paramType = ft->GetParameterType(i);
        // FIXME: this type conv should happen when we create the function
        // type!
        Expr *d = TypeConvertExpr(paramDefault, paramType, "function call default argument");
        if (d == nullptr)
            return nullptr;
        callargs.push_back(d);
    }

    // Now evaluate the values of all of the parameters being passed.
    std::vector<llvm::Value *> argVals;
    for (unsigned int i = 0; i < callargs.size(); ++i) {
        Expr *argExpr = callargs[i];
        if (argExpr == nullptr)
            // give up; we hit an error earlier
            return nullptr;

        llvm::Value *argValue = argExpr->GetValue(ctx);
        if (argValue == nullptr)
            // something went wrong in evaluating the argument's
            // expression, so give up on this
            return nullptr;

        argVals.push_back(argValue);
    }

    llvm::Value *retVal = nullptr;
    ctx->SetDebugPos(pos);
    if (ft->isTask) {
        AssertPos(pos, launchCountExpr[0] != nullptr);
        llvm::Value *launchCount[3] = {launchCountExpr[0]->GetValue(ctx), launchCountExpr[1]->GetValue(ctx),
                                       launchCountExpr[2]->GetValue(ctx)};

        if (launchCount[0] != nullptr)
            ctx->LaunchInst(callee, argVals, launchCount, ft);
    } else {
        if (isInvoke) {
            return ctx->InvokeSyclInst(callee, ft, argVals);
        } else {
            retVal = ctx->CallInst(callee, ft, argVals, isVoidFunc ? "" : "calltmp");
        }
    }

    if (isVoidFunc)
        return nullptr;
    else
        return retVal;
}

llvm::Value *FunctionCallExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (GetLValueType() != nullptr) {
        return GetValue(ctx);
    } else {
        // Only be a valid LValue type if the function
        // returns a pointer or reference.
        return nullptr;
    }
}

static bool lFullResolveOverloads(Expr *func, ExprList *args, std::vector<const Type *> *argTypes,
                                  std::vector<bool> *argCouldBeNULL, std::vector<bool> *argIsConstant) {
    for (unsigned int i = 0; i < args->exprs.size(); ++i) {
        Expr *expr = args->exprs[i];
        if (expr == nullptr)
            return false;
        const Type *t = expr->GetType();
        if (t == nullptr)
            return false;
        argTypes->push_back(t);
        argCouldBeNULL->push_back(lIsAllIntZeros(expr) || llvm::dyn_cast<NullPointerExpr>(expr));
        argIsConstant->push_back(llvm::dyn_cast<ConstExpr>(expr) || llvm::dyn_cast<NullPointerExpr>(expr));
    }
    return true;
}

const Type *FunctionCallExpr::GetType() const {
    std::vector<const Type *> argTypes;
    std::vector<bool> argCouldBeNULL, argIsConstant;
    if (func == nullptr || args == nullptr)
        return nullptr;

    if (lFullResolveOverloads(func, args, &argTypes, &argCouldBeNULL, &argIsConstant) == true) {
        FunctionSymbolExpr *fse = llvm::dyn_cast<FunctionSymbolExpr>(func);
        if (fse != nullptr) {
            fse->ResolveOverloads(args->pos, argTypes, &argCouldBeNULL, &argIsConstant);
        }
    }
    const Type *type = lGetFunctionType(func);
    if (type && type->IsDependentType()) {
        return type;
    }
    const FunctionType *ftype = CastType<FunctionType>(type);
    return ftype ? ftype->GetReturnType() : nullptr;
}

const Type *FunctionCallExpr::GetLValueType() const {
    const Type *type = lGetFunctionType(func);
    if (type->IsDependentType()) {
        return nullptr;
    }
    const FunctionType *ftype = CastType<FunctionType>(type);
    if (ftype && (ftype->GetReturnType()->IsPointerType() || ftype->GetReturnType()->IsReferenceType())) {
        return ftype->GetReturnType();
    } else {
        // Only be a valid LValue type if the function
        // returns a pointer or reference.
        return nullptr;
    }
}

Expr *FunctionCallExpr::Optimize() {
    if (func == nullptr || args == nullptr)
        return nullptr;
    return this;
}

Expr *FunctionCallExpr::TypeCheck() {
    if (func == nullptr || args == nullptr)
        return nullptr;

    std::vector<const Type *> argTypes;
    std::vector<bool> argCouldBeNULL, argIsConstant;

    if (lFullResolveOverloads(func, args, &argTypes, &argCouldBeNULL, &argIsConstant) == false) {
        return nullptr;
    }

    FunctionSymbolExpr *fse = llvm::dyn_cast<FunctionSymbolExpr>(func);
    const FunctionType *funcType = nullptr;
    if (fse != nullptr) {
        // Regular function call
        if (fse->ResolveOverloads(args->pos, argTypes, &argCouldBeNULL, &argIsConstant) == false)
            return nullptr;

        func = ::TypeCheck(fse);
        if (func == nullptr)
            return nullptr;

        funcType = CastType<FunctionType>(func->GetType());
        if (funcType == nullptr) {
            const PointerType *pt = CastType<PointerType>(func->GetType());
            funcType = (pt == nullptr) ? nullptr : CastType<FunctionType>(pt->GetBaseType());
        }

        if (funcType == nullptr) {
            Error(pos, "Valid function name must be used for function call.");
            return nullptr;
        }
    } else {
        // Call through a function pointer
        const Type *fptrType = func->GetType();
        if (fptrType == nullptr)
            return nullptr;

        // Make sure we do in fact have a function to call
        if (CastType<PointerType>(fptrType) == nullptr ||
            (funcType = CastType<FunctionType>(fptrType->GetBaseType())) == nullptr) {
            Error(func->pos, "Must provide function name or function pointer for "
                             "function call expression.");
            return nullptr;
        }

        // Make sure we don't have too many arguments for the function
        if ((int)argTypes.size() > funcType->GetNumParameters()) {
            Error(args->pos,
                  "Too many parameter values provided in "
                  "function call (%d provided, %d expected).",
                  (int)argTypes.size(), funcType->GetNumParameters());
            return nullptr;
        }
        // It's ok to have too few arguments, as long as the function's
        // default parameter values have started by the time we run out
        // of arguments
        if ((int)argTypes.size() < funcType->GetNumParameters() &&
            funcType->GetParameterDefault(argTypes.size()) == nullptr) {
            Error(args->pos,
                  "Too few parameter values provided in "
                  "function call (%d provided, %d expected).",
                  (int)argTypes.size(), funcType->GetNumParameters());
            return nullptr;
        }

        // Now make sure they can all type convert to the corresponding
        // parameter types..
        for (int i = 0; i < (int)argTypes.size(); ++i) {
            if (i < funcType->GetNumParameters()) {
                // make sure it can type convert
                const Type *paramType = funcType->GetParameterType(i);
                if (CanConvertTypes(argTypes[i], paramType) == false &&
                    !(argCouldBeNULL[i] == true && CastType<PointerType>(paramType) != nullptr)) {
                    Error(args->exprs[i]->pos,
                          "Can't convert argument of "
                          "type \"%s\" to type \"%s\" for function call "
                          "argument.",
                          argTypes[i]->GetString().c_str(), paramType->GetString().c_str());
                    return nullptr;
                }
            } else
                // Otherwise the parameter default saves us.  It should
                // be there for sure, given the check right above the
                // for loop.
                AssertPos(pos, funcType->GetParameterDefault(i) != nullptr);
        }

        if (fptrType->IsVaryingType()) {
            const Type *retType = funcType->GetReturnType();
            if (retType->IsVoidType() == false && retType->IsUniformType()) {
                Error(pos,
                      "Illegal to call a varying function pointer that "
                      "points to a function with a uniform return type \"%s\".",
                      funcType->GetReturnType()->GetString().c_str());
                return nullptr;
            }
        }
    }

    if (funcType->isTask) {
        if (!isLaunch)
            Error(pos, "\"launch\" expression needed to call function "
                       "with \"task\" qualifier.");
        for (int k = 0; k < 3; k++) {
            if (!launchCountExpr[k])
                return nullptr;

            launchCountExpr[k] = TypeConvertExpr(launchCountExpr[k], AtomicType::UniformInt32, "task launch count");
            if (launchCountExpr[k] == nullptr)
                return nullptr;
        }
    } else {
        if (isLaunch) {
            Error(pos, "\"launch\" expression illegal with non-\"task\"-"
                       "qualified function.");
            return nullptr;
        }
        AssertPos(pos, launchCountExpr[0] == nullptr);
    }
    if (isInvoke && !funcType->isExternSYCL) {
        Error(pos, "\"invoke_sycl\" expression illegal with non-\'extern \"SYCL\"\'-"
                   "qualified function.");
        return nullptr;
    }

    if (isInvoke && !funcType->isRegCall) {
        Error(pos, "\"invoke_sycl\" expression can be only used with \'__regcall\'-"
                   "qualified function.");
        return nullptr;
    }

    if (!isInvoke && funcType->isExternSYCL) {
        Error(pos, "Illegal to call \'extern \"SYCL\"\'-qualified function without \"invoke_sycl\" expression.");
        return nullptr;
    }

    if (func == nullptr || args == nullptr)
        return nullptr;
    return this;
}

int FunctionCallExpr::EstimateCost() const {
    if (isLaunch)
        return COST_TASK_LAUNCH;
    if (isInvoke)
        return COST_INVOKE;

    const Type *type = func->GetType();
    if (type == nullptr)
        return 0;

    const PointerType *pt = CastType<PointerType>(type);
    if (pt != nullptr)
        type = type->GetBaseType();

    const FunctionType *ftype = CastType<FunctionType>(type);
    if (ftype != nullptr && ftype->costOverride > -1)
        return ftype->costOverride;

    if (pt != nullptr)
        return pt->IsUniformType() ? COST_FUNPTR_UNIFORM : COST_FUNPTR_VARYING;
    else
        return COST_FUNCALL;
}

FunctionCallExpr *FunctionCallExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instFunc = func ? func->Instantiate(templInst) : nullptr;
    ExprList *instantiatedArgs = args ? args->Instantiate(templInst) : nullptr;
    FunctionCallExpr *inst = new FunctionCallExpr(instFunc, instantiatedArgs, pos, isLaunch);
    inst->launchCountExpr[0] = launchCountExpr[0] ? launchCountExpr[0]->Instantiate(templInst) : nullptr;
    inst->launchCountExpr[1] = launchCountExpr[1] ? launchCountExpr[1]->Instantiate(templInst) : nullptr;
    inst->launchCountExpr[2] = launchCountExpr[2] ? launchCountExpr[2]->Instantiate(templInst) : nullptr;
    return inst;
}

void FunctionCallExpr::Print(Indent &indent) const {
    if (!func || !args || !GetType()) {
        indent.Print("FunctionCallExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }
    indent.Print("FunctionCallExpr", pos);

    printf("[%s] %s %s\n", GetType()->GetString().c_str(), isLaunch ? "launch" : "", isInvoke ? "invoke_sycl" : "");
    indent.pushList(2);
    indent.setNextLabel("func");
    func->Print(indent);
    indent.setNextLabel("args");
    args->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// ExprList

bool ExprList::HasAmbiguousVariability(std::vector<const Expr *> &warn) const {
    bool hasAmbiguousVariability = false;
    for (unsigned int i = 0; i < exprs.size(); ++i) {
        if (exprs[i] != nullptr) {
            hasAmbiguousVariability |= exprs[i]->HasAmbiguousVariability(warn);
        }
    }
    return hasAmbiguousVariability;
}

llvm::Value *ExprList::GetValue(FunctionEmitContext *ctx) const {
    FATAL("ExprList::GetValue() should never be called");
    return nullptr;
}

const Type *ExprList::GetType() const {
    FATAL("ExprList::GetType() should never be called");
    return nullptr;
}

ExprList *ExprList::Optimize() { return this; }

ExprList *ExprList::TypeCheck() { return this; }

static std::pair<llvm::Constant *, bool> lGetExprListConstant(const Type *type, const ExprList *eList,
                                                              bool isStorageType) {
    std::vector<Expr *> exprs = eList->exprs;
    SourcePos pos = eList->pos;
    bool isVaryingInit = false;
    bool isNotValidForMultiTargetGlobal = false;
    if (exprs.size() == 1 && (CastType<AtomicType>(type) != nullptr || CastType<EnumType>(type) != nullptr ||
                              CastType<PointerType>(type) != nullptr)) {
        if (isStorageType)
            return exprs[0]->GetStorageConstant(type);
        else
            return exprs[0]->GetConstant(type);
    }

    const CollectionType *collectionType = CastType<CollectionType>(type);
    if (collectionType == nullptr) {
        if (type->IsVaryingType() == true) {
            isVaryingInit = true;
        } else
            return std::pair<llvm::Constant *, bool>(nullptr, false);
    }

    std::string name;
    if (CastType<StructType>(type) != nullptr)
        name = "struct";
    else if (CastType<ArrayType>(type) != nullptr)
        name = "array";
    else if (CastType<VectorType>(type) != nullptr)
        name = "vector";
    else if (isVaryingInit == true)
        name = "varying";
    else
        FATAL("Unexpected CollectionType in lGetExprListConstant");

    int elementCount = (isVaryingInit == true) ? g->target->getVectorWidth() : collectionType->GetElementCount();
    if ((int)exprs.size() > elementCount) {
        const Type *errType = (isVaryingInit == true) ? type : collectionType;
        Error(pos,
              "Initializer list for %s \"%s\" must have no more than %d "
              "elements (has %d).",
              name.c_str(), errType->GetString().c_str(), elementCount, (int)exprs.size());
        return std::pair<llvm::Constant *, bool>(nullptr, false);
    } else if ((isVaryingInit == true) && ((int)exprs.size() < elementCount)) {
        Error(pos,
              "Initializer list for %s \"%s\" must have %d "
              "elements (has %d).",
              name.c_str(), type->GetString().c_str(), elementCount, (int)exprs.size());
        return std::pair<llvm::Constant *, bool>(nullptr, false);
    }

    std::vector<llvm::Constant *> cv;
    for (unsigned int i = 0; i < exprs.size(); ++i) {
        if (exprs[i] == nullptr)
            return std::pair<llvm::Constant *, bool>(nullptr, false);
        const Type *elementType =
            (isVaryingInit == true) ? type->GetAsUniformType() : collectionType->GetElementType(i);

        Expr *expr = exprs[i];

        if (llvm::dyn_cast<ExprList>(expr) == nullptr) {
            // If there's a simple type conversion from the type of this
            // expression to the type we need, then let the regular type
            // conversion machinery handle it.
            expr = TypeConvertExpr(exprs[i], elementType, "initializer list");
            if (expr == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                return std::pair<llvm::Constant *, bool>(nullptr, false);
            }
            // Re-establish const-ness if possible
            expr = ::Optimize(expr);
        }
        std::pair<llvm::Constant *, bool> cPair;
        if (isStorageType)
            cPair = expr->GetStorageConstant(elementType);
        else
            cPair = expr->GetConstant(elementType);
        llvm::Constant *c = cPair.first;
        if (c == nullptr)
            // If this list element couldn't convert to the right constant
            // type for the corresponding collection member, then give up.
            return std::pair<llvm::Constant *, bool>(nullptr, false);
        isNotValidForMultiTargetGlobal = isNotValidForMultiTargetGlobal || cPair.second;
        cv.push_back(c);
    }

    // If there are too few, then treat missing ones as if they were zero
    if (isVaryingInit == false) {
        for (int i = (int)exprs.size(); i < collectionType->GetElementCount(); ++i) {
            const Type *elementType = collectionType->GetElementType(i);
            if (elementType == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                return std::pair<llvm::Constant *, bool>(nullptr, false);
            }
            llvm::Type *llvmType = elementType->LLVMType(g->ctx);
            if (llvmType == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                return std::pair<llvm::Constant *, bool>(nullptr, false);
            }

            llvm::Constant *c = llvm::Constant::getNullValue(llvmType);
            cv.push_back(c);
        }
    }

    if (CastType<StructType>(type) != nullptr) {
        llvm::StructType *llvmStructType = llvm::dyn_cast<llvm::StructType>(collectionType->LLVMType(g->ctx));
        AssertPos(pos, llvmStructType != nullptr);
        return std::pair<llvm::Constant *, bool>(llvm::ConstantStruct::get(llvmStructType, cv),
                                                 isNotValidForMultiTargetGlobal);
    } else {
        llvm::Type *lt = type->LLVMType(g->ctx);
        llvm::ArrayType *lat = llvm::dyn_cast<llvm::ArrayType>(lt);
        if (lat != nullptr)
            return std::pair<llvm::Constant *, bool>(llvm::ConstantArray::get(lat, cv), isNotValidForMultiTargetGlobal);
        else if (type->IsVaryingType()) {
            // uniform short vector type
            llvm::VectorType *lvt = llvm::dyn_cast<llvm::VectorType>(lt);
            AssertPos(pos, lvt != nullptr);
            int vectorWidth = g->target->getVectorWidth();

            while ((cv.size() % vectorWidth) != 0) {
                cv.push_back(llvm::UndefValue::get(lvt->getElementType()));
            }

            return std::pair<llvm::Constant *, bool>(llvm::ConstantVector::get(cv), isNotValidForMultiTargetGlobal);
        } else {
            // uniform short vector type
            AssertPos(pos, type->IsUniformType() && CastType<VectorType>(type) != nullptr);

            llvm::VectorType *lvt = llvm::dyn_cast<llvm::VectorType>(lt);
            AssertPos(pos, lvt != nullptr);

            // Uniform short vectors are stored as vectors of length
            // rounded up to a power of 2 bits in size but not less then 128 bit.
            // So we add additional undef values here until we get the right size.
            const VectorType *vt = CastType<VectorType>(type);
            int vectorWidth = vt->getVectorMemoryCount();

            while ((cv.size() % vectorWidth) != 0) {
                cv.push_back(llvm::UndefValue::get(lvt->getElementType()));
            }

            return std::pair<llvm::Constant *, bool>(llvm::ConstantVector::get(cv), isNotValidForMultiTargetGlobal);
        }
    }
}

std::pair<llvm::Constant *, bool> ExprList::GetStorageConstant(const Type *type) const {
    return lGetExprListConstant(type, this, true);
}
std::pair<llvm::Constant *, bool> ExprList::GetConstant(const Type *type) const {
    return lGetExprListConstant(type, this, false);
}

int ExprList::EstimateCost() const { return 0; }

ExprList *ExprList::Instantiate(TemplateInstantiation &templInst) const {
    ExprList *inst = new ExprList(pos);
    for (auto e : exprs) {
        inst->exprs.push_back(e->Instantiate(templInst));
    }
    return inst;
}

void ExprList::Print(Indent &indent) const {
    indent.PrintLn("ExprList", pos);

    indent.pushList(exprs.size());
    for (unsigned int i = 0; i < exprs.size(); ++i) {
        if (exprs[i] != nullptr) {
            exprs[i]->Print(indent);
        } else {
            indent.Print("<NULL>");
            indent.Done();
        }
    }

    indent.Done();
}

bool ExprList::HasAtomicInitializerList(std::map<AtomicType::BasicType, std::vector<ExprPosMapping>> &map) {
    bool isAtomicInit = true;

    // Go through all initializer expressions and check if they are atomic
    for (int i = 0; i < exprs.size(); ++i) {
        const AtomicType *at = CastType<AtomicType>(exprs[i]->GetType());
        if (at) {
            map[at->basicType].push_back({exprs[i], i});
        } else {
            isAtomicInit = false;
            break;
        }
    }

    return isAtomicInit;
}

///////////////////////////////////////////////////////////////////////////
// IndexExpr

IndexExpr::IndexExpr(Expr *a, Expr *i, SourcePos p) : Expr(p, IndexExprID) {
    baseExpr = a;
    index = i;
    type = lvalueType = nullptr;
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
static llvm::Value *lAddVaryingOffsetsIfNeeded(FunctionEmitContext *ctx, llvm::Value *ptr, const Type *ptrRefType) {
    if (CastType<ReferenceType>(ptrRefType) != nullptr)
        // References are uniform pointers, so no offsetting is needed
        return ptr;

    const PointerType *ptrType = CastType<PointerType>(ptrRefType);
    Assert(ptrType != nullptr);
    if (ptrType->IsUniformType() || ptrType->IsSlice())
        return ptr;

    const Type *baseType = ptrType->GetBaseType();
    if (baseType->IsVaryingType() == false)
        return ptr;

    // must be indexing into varying atomic, enum, or pointer types
    if (Type::IsBasicType(baseType) == false)
        return ptr;

    // Onward: compute the per lane offsets.
    llvm::Value *varyingOffsets = ctx->ProgramIndexVector();

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
static bool lVaryingStructHasUniformMember(const Type *type, SourcePos pos) {
    if (CastType<VectorType>(type) != nullptr || CastType<ReferenceType>(type) != nullptr)
        return false;

    const StructType *st = CastType<StructType>(type);
    if (st == nullptr) {
        const ArrayType *at = CastType<ArrayType>(type);
        if (at != nullptr)
            st = CastType<StructType>(at->GetElementType());
        else {
            const PointerType *pt = CastType<PointerType>(type);
            if (pt == nullptr)
                return false;

            st = CastType<StructType>(pt->GetBaseType());
        }

        if (st == nullptr)
            return false;
    }

    if (st->IsVaryingType() == false)
        return false;

    for (int i = 0; i < st->GetElementCount(); ++i) {
        const Type *eltType = st->GetElementType(i);
        if (eltType == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            continue;
        }

        if (CastType<StructType>(eltType) != nullptr) {
            // We know that the enclosing struct is varying at this point,
            // so push that down to the enclosed struct before makign the
            // recursive call.
            eltType = eltType->GetAsVaryingType();
            if (lVaryingStructHasUniformMember(eltType, pos))
                return true;
        } else if (eltType->IsUniformType()) {
            Error(pos,
                  "Gather operation is impossible due to the presence of "
                  "struct member \"%s\" with uniform type \"%s\" in the "
                  "varying struct type \"%s\".",
                  st->GetElementName(i).c_str(), eltType->GetString().c_str(), st->GetString().c_str());
            return true;
        }
    }

    return false;
}

llvm::Value *IndexExpr::GetValue(FunctionEmitContext *ctx) const {
    const Type *indexType, *returnType;
    if (baseExpr == nullptr || index == nullptr || ((indexType = index->GetType()) == nullptr) ||
        ((returnType = GetType()) == nullptr)) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    // If this is going to be a gather, make sure that the varying return
    // type can represent the result (i.e. that we don't have a bound
    // 'uniform' member in a varying struct...)
    if (indexType->IsVaryingType() && lVaryingStructHasUniformMember(returnType, pos))
        return nullptr;

    ctx->SetDebugPos(pos);

    llvm::Value *ptr = GetLValue(ctx);
    llvm::Value *mask = nullptr;
    const Type *lvType = GetLValueType();
    if (ptr == nullptr) {
        // We may be indexing into a temporary that hasn't hit memory, so
        // get the full value and stuff it into temporary alloca'd space so
        // that we can index from there...
        const Type *baseExprType = baseExpr->GetType();
        llvm::Value *val = baseExpr->GetValue(ctx);
        if (baseExprType == nullptr || val == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }
        ctx->SetDebugPos(pos);
        AddressInfo *tmpPtrInfo = ctx->AllocaInst(baseExprType, "array_tmp");
        ctx->StoreInst(val, tmpPtrInfo, baseExprType);

        // Get a pointer type to the underlying elements
        const SequentialType *st = CastType<SequentialType>(baseExprType);
        if (st == nullptr) {
            Assert(m->errorCount > 0);
            return nullptr;
        }
        lvType = PointerType::GetUniform(st->GetElementType());

        // And do the indexing calculation into the temporary array in memory
        ptr = ctx->GetElementPtrInst(tmpPtrInfo->getPointer(), LLVMInt32(0), index->GetValue(ctx),
                                     PointerType::GetUniform(baseExprType));
        ptr = lAddVaryingOffsetsIfNeeded(ctx, ptr, lvType);

        mask = LLVMMaskAllOn;
    } else {
        Symbol *baseSym = GetBaseSymbol();
        if (llvm::dyn_cast<FunctionCallExpr>(baseExpr) == nullptr && llvm::dyn_cast<BinaryExpr>(baseExpr) == nullptr &&
            llvm::dyn_cast<SelectExpr>(baseExpr) == nullptr) {
            // Don't check if we're doing a function call or pointer arith or select
            AssertPos(pos, baseSym != nullptr);
        }
        mask = lMaskForSymbol(baseSym, ctx);
    }

    ctx->SetDebugPos(pos);
    return ctx->LoadInst(ptr, mask, lvType);
}

const Type *IndexExpr::GetType() const {
    if (type != nullptr)
        return type;

    const Type *baseExprType, *indexType;
    if (!baseExpr || !index || ((baseExprType = baseExpr->GetType()) == nullptr) ||
        ((indexType = index->GetType()) == nullptr))
        return nullptr;

    if (baseExprType->IsDependentType() || indexType->IsDependentType()) {
        return AtomicType::Dependent;
    }

    const Type *elementType = nullptr;
    const PointerType *pointerType = CastType<PointerType>(baseExprType);
    if (pointerType != nullptr)
        // ptr[index] -> type that the pointer points to
        elementType = pointerType->GetBaseType();
    else if (const SequentialType *sequentialType = CastType<SequentialType>(baseExprType->GetReferenceTarget()))
        // sequential type[index] -> element type of the sequential type
        elementType = sequentialType->GetElementType();
    else
        // Not an expression that can be indexed into. Will result in error.
        return nullptr;

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
    if (indexType->IsUniformType() && (pointerType == nullptr || pointerType->IsUniformType()))
        type = elementType;
    else
        type = elementType->GetAsVaryingType();

    return type;
}

Symbol *IndexExpr::GetBaseSymbol() const { return baseExpr ? baseExpr->GetBaseSymbol() : nullptr; }

/** Utility routine that takes a regualr pointer (either uniform or
    varying) and returns a slice pointer with zero offsets.
 */
static llvm::Value *lConvertToSlicePointer(FunctionEmitContext *ctx, llvm::Value *ptr,
                                           const PointerType *slicePtrType) {
    llvm::Type *llvmSlicePtrType = slicePtrType->LLVMType(g->ctx);
    llvm::StructType *sliceStructType = llvm::dyn_cast<llvm::StructType>(llvmSlicePtrType);
    Assert(sliceStructType != nullptr && sliceStructType->getElementType(0) == ptr->getType());

    // Get a null-initialized struct to take care of having zeros for the
    // offsets
    llvm::Value *result = llvm::Constant::getNullValue(sliceStructType);
    // And replace the pointer in the struct with the given pointer
    return ctx->InsertInst(result, ptr, 0, llvm::Twine(ptr->getName()) + "_slice");
}

/** If the given array index is a compile time constant, check to see if it
    value/values don't go past the end of the array; issue a warning if
    so.
*/
static void lCheckIndicesVersusBounds(const Type *baseExprType, Expr *index) {
    const SequentialType *seqType = CastType<SequentialType>(baseExprType);
    if (seqType == nullptr)
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

    ConstExpr *ce = llvm::dyn_cast<ConstExpr>(index);
    if (ce == nullptr)
        return;

    int32_t indices[ISPC_MAX_NVEC];
    int count = ce->GetValues(indices);
    for (int i = 0; i < count; ++i) {
        if (indices[i] < 0 || indices[i] >= nElements)
            Warning(index->pos,
                    "Array index \"%d\" may be out of bounds for %d "
                    "element array.",
                    indices[i], nElements);
    }
}

/** Converts the given pointer value to a slice pointer if the pointer
    points to SOA'ed data.
*/
static llvm::Value *lConvertPtrToSliceIfNeeded(FunctionEmitContext *ctx, llvm::Value *ptr, const Type **type) {
    Assert(*type != nullptr);
    const PointerType *ptrType = CastType<PointerType>(*type);
    Assert(ptrType != nullptr);
    bool convertToSlice = (ptrType->GetBaseType()->IsSOAType() && ptrType->IsSlice() == false);
    if (convertToSlice == false)
        return ptr;

    *type = ptrType->GetAsSlice();
    return lConvertToSlicePointer(ctx, ptr, ptrType->GetAsSlice());
}

llvm::Value *IndexExpr::GetLValue(FunctionEmitContext *ctx) const {
    const Type *baseExprType;
    if (baseExpr == nullptr || index == nullptr || ((baseExprType = baseExpr->GetType()) == nullptr)) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    ctx->SetDebugPos(pos);
    llvm::Value *indexValue = index->GetValue(ctx);
    if (indexValue == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    ctx->SetDebugPos(pos);
    if (CastType<PointerType>(baseExprType) != nullptr) {
        // We're indexing off of a pointer
        llvm::Value *basePtrValue = baseExpr->GetValue(ctx);
        if (basePtrValue == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }
        ctx->SetDebugPos(pos);

        // Convert to a slice pointer if we're indexing into SOA data
        basePtrValue = lConvertPtrToSliceIfNeeded(ctx, basePtrValue, &baseExprType);

        llvm::Value *ptr = ctx->GetElementPtrInst(basePtrValue, indexValue, baseExprType,
                                                  llvm::Twine(basePtrValue->getName()) + "_offset");
        return lAddVaryingOffsetsIfNeeded(ctx, ptr, GetLValueType());
    }

    // Not a pointer: we must be indexing an array or vector (and possibly
    // a reference thereuponfore.)
    llvm::Value *basePtr = nullptr;
    const PointerType *basePtrType = nullptr;
    if (CastType<ArrayType>(baseExprType) || CastType<VectorType>(baseExprType)) {
        basePtr = baseExpr->GetLValue(ctx);
        basePtrType = CastType<PointerType>(baseExpr->GetLValueType());
        if (baseExpr->GetLValueType())
            AssertPos(pos, basePtrType != nullptr);
    } else {
        baseExprType = baseExprType->GetReferenceTarget();
        AssertPos(pos, CastType<ArrayType>(baseExprType) || CastType<VectorType>(baseExprType));
        basePtr = baseExpr->GetValue(ctx);
        basePtrType = PointerType::GetUniform(baseExprType);
    }
    if (!basePtr)
        return nullptr;

    // If possible, check the index value(s) against the size of the array
    lCheckIndicesVersusBounds(baseExprType, index);

    // Convert to a slice pointer if indexing into SOA data
    basePtr = lConvertPtrToSliceIfNeeded(ctx, basePtr, (const Type **)&basePtrType);

    ctx->SetDebugPos(pos);

    // And do the actual indexing calculation..
    llvm::Value *ptr = ctx->GetElementPtrInst(basePtr, LLVMInt32(0), indexValue, basePtrType,
                                              llvm::Twine(basePtr->getName()) + "_offset");
    return lAddVaryingOffsetsIfNeeded(ctx, ptr, GetLValueType());
}

const Type *IndexExpr::GetLValueType() const {
    if (lvalueType != nullptr)
        return lvalueType;

    const Type *baseExprType, *baseExprLValueType, *indexType;
    if (baseExpr == nullptr || index == nullptr || ((baseExprType = baseExpr->GetType()) == nullptr) ||
        ((baseExprLValueType = baseExpr->GetLValueType()) == nullptr) || ((indexType = index->GetType()) == nullptr))
        return nullptr;

    // regularize to a PointerType
    if (CastType<ReferenceType>(baseExprLValueType) != nullptr) {
        const Type *refTarget = baseExprLValueType->GetReferenceTarget();
        baseExprLValueType = PointerType::GetUniform(refTarget);
    }
    AssertPos(pos, CastType<PointerType>(baseExprLValueType) != nullptr);

    // Find the type of thing that we're indexing into
    const Type *elementType;
    const SequentialType *st = CastType<SequentialType>(baseExprLValueType->GetBaseType());
    if (st != nullptr)
        elementType = st->GetElementType();
    else {
        const PointerType *pt = CastType<PointerType>(baseExprLValueType->GetBaseType());
        // This assertion seems overly strict.
        // Why does it need to be a pointer to a pointer?
        // AssertPos(pos, pt != nullptr);

        if (pt != nullptr) {
            elementType = pt->GetBaseType();
        } else {
            elementType = baseExprLValueType->GetBaseType();
        }
    }

    // Are we indexing into a varying type, or are we indexing with a
    // varying pointer?
    bool baseVarying;
    if (CastType<PointerType>(baseExprType) != nullptr)
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

Expr *IndexExpr::Optimize() {
    if (baseExpr == nullptr || index == nullptr)
        return nullptr;
    return this;
}

Expr *IndexExpr::TypeCheck() {
    const Type *indexType;
    if (baseExpr == nullptr || index == nullptr || ((indexType = index->GetType()) == nullptr)) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    const Type *baseExprType = baseExpr->GetType();
    if (baseExprType == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (baseExprType->IsDependentType() || indexType->IsDependentType()) {
        return this;
    }

    if (!CastType<SequentialType>(baseExprType->GetReferenceTarget())) {
        if (const PointerType *pt = CastType<PointerType>(baseExprType)) {
            if (pt->GetBaseType()->IsVoidType()) {
                Error(pos, "Illegal to dereference void pointer type \"%s\".", baseExprType->GetString().c_str());
                return nullptr;
            }
        } else {
            Error(pos,
                  "Trying to index into non-array, vector, or pointer "
                  "type \"%s\".",
                  baseExprType->GetString().c_str());
            return nullptr;
        }
    }

    bool isUniform = (index->GetType()->IsUniformType() && !g->opt.disableUniformMemoryOptimizations);

    if (!isUniform) {
        // Unless we have an explicit 64-bit index and are compiling to a
        // 64-bit target with 64-bit addressing, convert the index to an int32
        // type.
        //    The range of varying index is limited to [0,2^31) as a result.
        if (!(Type::EqualIgnoringConst(indexType->GetAsUniformType(), AtomicType::UniformUInt64) ||
              Type::EqualIgnoringConst(indexType->GetAsUniformType(), AtomicType::UniformInt64)) ||
            g->target->is32Bit() || g->opt.force32BitAddressing) {
            const Type *indexType = AtomicType::VaryingInt32;
            index = TypeConvertExpr(index, indexType, "array index");
            if (index == nullptr)
                return nullptr;
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
        bool force_32bit =
            g->target->is32Bit() || (g->opt.force32BitAddressing &&
                                     Type::EqualIgnoringConst(indexType->GetAsUniformType(), AtomicType::UniformInt64));
        const Type *indexType = force_32bit ? AtomicType::UniformInt32 : AtomicType::UniformInt64;
        index = TypeConvertExpr(index, indexType, "array index");
        if (index == nullptr)
            return nullptr;
    }

    return this;
}

int IndexExpr::EstimateCost() const {
    if (index == nullptr || baseExpr == nullptr)
        return 0;

    const Type *indexType = index->GetType();
    const Type *baseExprType = baseExpr->GetType();

    if ((indexType != nullptr && indexType->IsVaryingType()) ||
        (CastType<PointerType>(baseExprType) != nullptr && baseExprType->IsVaryingType()))
        // be pessimistic; some of these will later turn out to be vector
        // loads/stores, but it's too early for us to know that here.
        return COST_GATHER;
    else
        return COST_LOAD;
}

IndexExpr *IndexExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instBaseExpr = baseExpr ? baseExpr->Instantiate(templInst) : nullptr;
    Expr *instIndex = index ? index->Instantiate(templInst) : nullptr;
    return new IndexExpr(instBaseExpr, instIndex, pos);
}

void IndexExpr::Print(Indent &indent) const {
    if (!baseExpr || !index || !GetType()) {
        indent.Print("IndexExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("IndexExpr", pos);

    printf("[%s]\n", GetType()->GetString().c_str());
    indent.pushList(2);
    baseExpr->Print(indent);
    index->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// MemberExpr

/** Map one character ids to vector element numbers.  Allow a few different
    conventions--xyzw, rgba, uv.
*/
static int lIdentifierToVectorElement(char id) {
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

StructMemberExpr::StructMemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue)
    : MemberExpr(e, id, p, idpos, derefLValue, StructMemberExprID) {}

const Type *StructMemberExpr::GetType() const {
    if (type != nullptr)
        return type;

    // It's a struct, and the result type is the element type, possibly
    // promoted to varying if the struct type / lvalue is varying.
    const Type *exprType, *lvalueType;
    const StructType *structType;
    if (expr == nullptr || ((exprType = expr->GetType()) == nullptr) || ((structType = getStructType()) == nullptr) ||
        ((lvalueType = GetLValueType()) == nullptr)) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (exprType->IsDependentType() || structType->IsDependentType() || lvalueType->IsDependentType()) {
        return AtomicType::Dependent;
    }

    const Type *elementType = structType->GetElementType(identifier);
    if (elementType == nullptr) {
        Error(identifierPos, "Element name \"%s\" not present in struct type \"%s\".%s", identifier.c_str(),
              structType->GetString().c_str(), getCandidateNearMatches().c_str());
        return nullptr;
    }
    AssertPos(pos, Type::Equal(lvalueType->GetBaseType(), elementType));

    bool isSlice = (CastType<PointerType>(lvalueType) && CastType<PointerType>(lvalueType)->IsSlice());
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

const Type *StructMemberExpr::GetLValueType() const {
    if (lvalueType != nullptr)
        return lvalueType;

    if (expr == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    const Type *exprLValueType = dereferenceExpr ? expr->GetType() : expr->GetLValueType();
    if (exprLValueType == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    // The pointer type is varying if the lvalue type of the expression is
    // varying (and otherwise uniform)
    const PointerType *ptrType = (exprLValueType->IsUniformType() || CastType<ReferenceType>(exprLValueType) != nullptr)
                                     ? PointerType::GetUniform(getElementType())
                                     : PointerType::GetVarying(getElementType());

    // If struct pointer is a slice pointer, the resulting member pointer
    // needs to be a frozen slice pointer--i.e. any further indexing with
    // the result shouldn't modify the minor slice offset, but it should be
    // left unchanged until we get to a leaf SOA value.
    if (CastType<PointerType>(exprLValueType) && CastType<PointerType>(exprLValueType)->IsSlice())
        ptrType = ptrType->GetAsFrozenSlice();

    lvalueType = ptrType;
    return lvalueType;
}

int StructMemberExpr::getElementNumber() const {
    const StructType *structType = getStructType();
    if (structType == nullptr)
        return -1;

    int elementNumber = structType->GetElementNumber(identifier);
    if (elementNumber == -1)
        Error(identifierPos, "Element name \"%s\" not present in struct type \"%s\".%s", identifier.c_str(),
              structType->GetString().c_str(), getCandidateNearMatches().c_str());

    return elementNumber;
}

const Type *StructMemberExpr::getElementType() const {
    const StructType *structType = getStructType();
    if (structType == nullptr)
        return nullptr;

    return structType->GetElementType(identifier);
}

/** Returns the type of the underlying struct that we're returning a member
    of. */
const StructType *StructMemberExpr::getStructType() const {
    const Type *type = dereferenceExpr ? expr->GetType() : expr->GetLValueType();
    if (type == nullptr)
        return nullptr;

    const Type *structType;
    const ReferenceType *rt = CastType<ReferenceType>(type);
    if (rt != nullptr)
        structType = rt->GetReferenceTarget();
    else {
        const PointerType *pt = CastType<PointerType>(type);
        AssertPos(pos, pt != nullptr);
        structType = pt->GetBaseType();
    }

    const StructType *ret = CastType<StructType>(structType);
    AssertPos(pos, ret != nullptr);
    return ret;
}

//////////////////////////////////////////////////
// VectorMemberExpr

VectorMemberExpr::VectorMemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue)
    : MemberExpr(e, id, p, idpos, derefLValue, VectorMemberExprID) {
    const Type *exprType = e->GetType();
    exprVectorType = CastType<VectorType>(exprType);
    if (exprVectorType == nullptr) {
        const PointerType *pt = CastType<PointerType>(exprType);
        if (pt != nullptr)
            exprVectorType = CastType<VectorType>(pt->GetBaseType());
        else {
            AssertPos(pos, CastType<ReferenceType>(exprType) != nullptr);
            exprVectorType = CastType<VectorType>(exprType->GetReferenceTarget());
        }
        AssertPos(pos, exprVectorType != nullptr);
    }
    memberType = new VectorType(exprVectorType->GetElementType(), identifier.length());
}

const Type *VectorMemberExpr::GetType() const {
    if (type != nullptr)
        return type;

    // For 1-element expressions, we have the base vector element
    // type.  For n-element expressions, we have a shortvec type
    // with n > 1 elements.  This can be changed when we get
    // type<1> -> type conversions.
    const Type *t =
        (identifier.length() == 1) ? (const Type *)exprVectorType->GetElementType() : (const Type *)memberType;

    if (t->IsDependentType()) {
        return AtomicType::Dependent;
    }

    type = t;

    const Type *lvType = GetLValueType();
    if (lvType != nullptr) {
        bool isSlice = (CastType<PointerType>(lvType) && CastType<PointerType>(lvType)->IsSlice());
        if (isSlice) {
            // CO            AssertPos(pos, type->IsSOAType());
            if (lvType->IsUniformType())
                type = type->GetAsUniformType();
        }

        if (lvType->IsVaryingType())
            type = type->GetAsVaryingType();
    }

    return type;
}

llvm::Value *VectorMemberExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (identifier.length() == 1) {
        return MemberExpr::GetLValue(ctx);
    } else {
        return nullptr;
    }
}

const Type *VectorMemberExpr::GetLValueType() const {
    if (lvalueType != nullptr)
        return lvalueType;

    if (identifier.length() == 1) {
        if (expr == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }

        const Type *exprLValueType = dereferenceExpr ? expr->GetType() : expr->GetLValueType();
        if (exprLValueType == nullptr)
            return nullptr;

        const VectorType *vt = nullptr;
        if (CastType<ReferenceType>(exprLValueType) != nullptr)
            vt = CastType<VectorType>(exprLValueType->GetReferenceTarget());
        else
            vt = CastType<VectorType>(exprLValueType->GetBaseType());
        AssertPos(pos, vt != nullptr);

        // we don't want to report that it's e.g. a pointer to a float<1>,
        // but a pointer to a float, etc.
        const Type *elementType = vt->GetElementType();
        if (CastType<ReferenceType>(exprLValueType) != nullptr)
            lvalueType = new ReferenceType(elementType);
        else {
            const PointerType *ptrType = exprLValueType->IsUniformType() ? PointerType::GetUniform(elementType)
                                                                         : PointerType::GetVarying(elementType);
            // FIXME: replicated logic with structmemberexpr....
            if (CastType<PointerType>(exprLValueType) && CastType<PointerType>(exprLValueType)->IsSlice())
                ptrType = ptrType->GetAsFrozenSlice();
            lvalueType = ptrType;
        }
    }

    return lvalueType;
}

llvm::Value *VectorMemberExpr::GetValue(FunctionEmitContext *ctx) const {
    if (identifier.length() == 1) {
        return MemberExpr::GetValue(ctx);
    } else {
        std::vector<int> indices;

        for (size_t i = 0; i < identifier.size(); ++i) {
            int idx = lIdentifierToVectorElement(identifier[i]);
            if (idx == -1)
                Error(pos, "Invalid swizzle character '%c' in swizzle \"%s\".", identifier[i], identifier.c_str());

            indices.push_back(idx);
        }

        llvm::Value *basePtr = nullptr;
        AddressInfo *basePtrInfo = nullptr;
        const Type *basePtrType = nullptr;
        if (dereferenceExpr) {
            basePtr = expr->GetValue(ctx);
            basePtrType = expr->GetType();
        } else {
            basePtr = expr->GetLValue(ctx);
            basePtrType = expr->GetLValueType();
        }

        if (basePtr == nullptr || basePtrType == nullptr) {
            // Check that expression on the left side is a rvalue expression
            llvm::Value *exprValue = expr->GetValue(ctx);
            basePtrInfo = ctx->AllocaInst(expr->GetType());
            basePtrType = PointerType::GetUniform(exprVectorType);
            if (basePtrInfo == nullptr || basePtrType == nullptr) {
                AssertPos(pos, m->errorCount > 0);
                return nullptr;
            }
            basePtr = basePtrInfo->getPointer();
            AssertPos(pos, basePtr != nullptr);
            ctx->StoreInst(exprValue, basePtrInfo, expr->GetType());
        }

        // Allocate temporary memory to store the result
        AddressInfo *resultPtrInfo = ctx->AllocaInst(memberType, "vector_tmp");
        if (resultPtrInfo == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }

        // FIXME: we should be able to use the internal mask here according
        // to the same logic where it's used elsewhere
        llvm::Value *elementMask = ctx->GetFullMask();

        const PointerType *ptrType = ctx->RegularizePointer(basePtrType);
        const Type *elementPtrType = ptrType->IsUniformType()
                                         ? PointerType::GetUniform(exprVectorType->GetElementType())
                                         : PointerType::GetVarying(exprVectorType->GetElementType());
        ctx->SetDebugPos(pos);
        for (size_t i = 0; i < identifier.size(); ++i) {
            char idStr[2] = {identifier[i], '\0'};
            llvm::Value *elementPtr = ctx->AddElementOffset(new AddressInfo(basePtr, basePtrType), indices[i],
                                                            llvm::Twine(basePtr->getName()) + idStr);
            llvm::Value *elementValue = ctx->LoadInst(elementPtr, elementMask, elementPtrType);

            llvm::Value *ptmp =
                ctx->AddElementOffset(resultPtrInfo, i, llvm::Twine(resultPtrInfo->getPointer()->getName()) + idStr);
            ctx->StoreInst(elementValue, new AddressInfo(ptmp, exprVectorType->GetElementType()),
                           elementPtrType->GetBaseType());
        }

        return ctx->LoadInst(resultPtrInfo, memberType, llvm::Twine(basePtr->getName()) + "_swizzle");
    }
}

int VectorMemberExpr::getElementNumber() const {
    int elementNumber = lIdentifierToVectorElement(identifier[0]);
    if (elementNumber == -1)
        Error(pos, "Vector element identifier \"%s\" unknown.", identifier.c_str());
    return elementNumber;
}

const Type *VectorMemberExpr::getElementType() const { return memberType; }

//////////////////////////////////////////////////
// DependentMemberExpr

DependentMemberExpr::DependentMemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue)
    : MemberExpr(e, id, p, idpos, derefLValue, DependentMemberExprID) {
    Assert(e != nullptr);
    Assert(id != nullptr);
}

int DependentMemberExpr::getElementNumber() const { UNREACHABLE(); };
const Type *DependentMemberExpr::getElementType() const { UNREACHABLE(); };

//////////////////////////////////////////////////
// MemberExpr

MemberExpr *MemberExpr::create(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue) {
    // FIXME: we need to call TypeCheck() here so that we can call
    // e->GetType() in the following.  But really we just shouldn't try to
    // resolve this now but just have a generic MemberExpr type that
    // handles all cases so that this is unnecessary.
    e = ::TypeCheck(e);

    const Type *exprType;
    if (e == nullptr || (exprType = e->GetType()) == nullptr)
        return nullptr;

    if (exprType->IsDependentType()) {
        return new DependentMemberExpr(e, id, p, idpos, derefLValue);
    }

    const ReferenceType *referenceType = CastType<ReferenceType>(exprType);
    if (referenceType != nullptr) {
        e = new RefDerefExpr(e, e->pos);
        exprType = e->GetType();
        Assert(exprType != nullptr);
    }

    const PointerType *pointerType = CastType<PointerType>(exprType);
    if (pointerType != nullptr)
        exprType = pointerType->GetBaseType();

    if (derefLValue == true && pointerType == nullptr) {
        const Type *targetType = exprType->GetReferenceTarget();
        if (CastType<StructType>(targetType) != nullptr)
            Error(p,
                  "Member operator \"->\" can't be applied to non-pointer "
                  "type \"%s\".  Did you mean to use \".\"?",
                  exprType->GetString().c_str());
        else
            Error(p,
                  "Member operator \"->\" can't be applied to non-struct "
                  "pointer type \"%s\".",
                  exprType->GetString().c_str());
        return nullptr;
    }
    // For struct and short-vector, emit error if elements are accessed
    // incorrectly.
    if (derefLValue == false && pointerType != nullptr &&
        ((CastType<StructType>(pointerType->GetBaseType()) != nullptr) ||
         (CastType<VectorType>(pointerType->GetBaseType()) != nullptr))) {
        Error(p,
              "Member operator \".\" can't be applied to pointer "
              "type \"%s\".  Did you mean to use \"->\"?",
              exprType->GetString().c_str());
        return nullptr;
    }
    if (CastType<StructType>(exprType) != nullptr) {
        const StructType *st = CastType<StructType>(exprType);
        if (st->IsDefined()) {
            std::string elemName(id);
            const Type *elemType = st->GetElementType(elemName);
            if (elemType == nullptr) {
                Error(p, "'%s' has no member named \"%s\"", st->GetString().c_str(), id);
                return nullptr;
            }
            return new StructMemberExpr(e, id, p, idpos, derefLValue);
        } else {
            Error(p,
                  "Member operator \"%s\" can't be applied to declared "
                  "struct \"%s\" containing an undefined struct type.",
                  derefLValue ? "->" : ".", exprType->GetString().c_str());
            return nullptr;
        }
    } else if (CastType<VectorType>(exprType) != nullptr)
        return new VectorMemberExpr(e, id, p, idpos, derefLValue);
    else if (CastType<UndefinedStructType>(exprType)) {
        Error(p,
              "Member operator \"%s\" can't be applied to declared "
              "but not defined struct type \"%s\".",
              derefLValue ? "->" : ".", exprType->GetString().c_str());
        return nullptr;
    } else {
        Error(p,
              "Member operator \"%s\" can't be used with expression of "
              "\"%s\" type.",
              derefLValue ? "->" : ".", exprType->GetString().c_str());
        return nullptr;
    }
}

MemberExpr::MemberExpr(Expr *e, const char *id, SourcePos p, SourcePos idpos, bool derefLValue, unsigned scid)
    : Expr(p, scid), identifierPos(idpos) {
    expr = e;
    identifier = id;
    dereferenceExpr = derefLValue;
    type = lvalueType = nullptr;
}

llvm::Value *MemberExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr)
        return nullptr;

    llvm::Value *lvalue = GetLValue(ctx);
    const Type *lvalueType = GetLValueType();

    llvm::Value *mask = nullptr;
    if (lvalue == nullptr) {
        if (m->errorCount > 0)
            return nullptr;

        // As in the array case, this may be a temporary that hasn't hit
        // memory; get the full value and stuff it into a temporary array
        // so that we can index from there...
        llvm::Value *val = expr->GetValue(ctx);
        if (!val) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }
        ctx->SetDebugPos(pos);
        const Type *exprType = expr->GetType();
        AddressInfo *ptrInfo = ctx->AllocaInst(exprType, "struct_tmp");
        ctx->StoreInst(val, ptrInfo, exprType);

        int elementNumber = getElementNumber();
        if (elementNumber == -1)
            return nullptr;

        lvalue = ctx->AddElementOffset(ptrInfo, elementNumber);
        lvalueType = PointerType::GetUniform(GetType());
        mask = LLVMMaskAllOn;
    } else {
        Symbol *baseSym = GetBaseSymbol();
        AssertPos(pos, baseSym != nullptr);
        mask = lMaskForSymbol(baseSym, ctx);
    }

    ctx->SetDebugPos(pos);
    std::string suffix = std::string("_") + identifier;
    return ctx->LoadInst(lvalue, mask, lvalueType, llvm::Twine(lvalue->getName()) + suffix);
}

const Type *MemberExpr::GetType() const { return nullptr; }

Symbol *MemberExpr::GetBaseSymbol() const { return expr ? expr->GetBaseSymbol() : nullptr; }

int MemberExpr::getElementNumber() const { return -1; }

llvm::Value *MemberExpr::GetLValue(FunctionEmitContext *ctx) const {
    const Type *exprType;
    if (!expr || ((exprType = expr->GetType()) == nullptr))
        return nullptr;

    ctx->SetDebugPos(pos);
    llvm::Value *basePtr = dereferenceExpr ? expr->GetValue(ctx) : expr->GetLValue(ctx);
    if (!basePtr)
        return nullptr;

    int elementNumber = getElementNumber();
    if (elementNumber == -1)
        return nullptr;

    const Type *exprLValueType = dereferenceExpr ? exprType : expr->GetLValueType();
    ctx->SetDebugPos(pos);
    llvm::Value *ptr = ctx->AddElementOffset(new AddressInfo(basePtr, exprLValueType), elementNumber,
                                             basePtr->getName().str().c_str());
    if (ptr == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    ptr = lAddVaryingOffsetsIfNeeded(ctx, ptr, GetLValueType());

    return ptr;
}

Expr *MemberExpr::TypeCheck() { return expr ? this : nullptr; }

Expr *MemberExpr::Optimize() { return expr ? this : nullptr; }

int MemberExpr::EstimateCost() const {
    const Type *lvalueType = GetLValueType();
    if (lvalueType != nullptr && lvalueType->IsVaryingType())
        return COST_GATHER + COST_SIMPLE_ARITH_LOGIC_OP;
    else
        return COST_SIMPLE_ARITH_LOGIC_OP;
}

MemberExpr *MemberExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return MemberExpr::create(instExpr, identifier.c_str(), pos, identifierPos, dereferenceExpr);
}

void MemberExpr::Print(Indent &indent) const {
    if (getValueID() == StructMemberExprID) {
        indent.Print("StructMemberExpr", pos);
    } else if (getValueID() == VectorMemberExprID) {
        indent.Print("VectorMemberExpr", pos);
    } else if (getValueID() == DependentMemberExprID) {
        indent.Print("DependentMemberExpr", pos);
    } else {
        indent.Print("MemberExpr", pos);
    }

    if (!expr || !GetType()) {
        indent.Print(" <NULL EXPR>\n");
        indent.Done();
        return;
    }

    printf("[%s] %s %s\n", GetType()->GetString().c_str(), dereferenceExpr ? "->" : ".", identifier.c_str());
    indent.pushSingle();
    expr->Print(indent);

    indent.Done();
}

/** There is no structure member with the name we've got in "identifier".
    Use the approximate string matching routine to see if the identifier is
    a minor misspelling of one of the ones that is there.
 */
std::string MemberExpr::getCandidateNearMatches() const {
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
        if (i < alternates.size() - 1)
            ret += ", or ";
    }
    ret += "?";
    return ret;
}

///////////////////////////////////////////////////////////////////////////
// ConstExpr

ConstExpr::ConstExpr(const Type *t, int8_t i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt8->GetAsConstType()));
    int8Val[0] = i;
}

ConstExpr::ConstExpr(const Type *t, int8_t *i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt8->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingInt8->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int8Val[j] = i[j];
}

ConstExpr::ConstExpr(const Type *t, uint8_t u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt8->GetAsConstType()));
    uint8Val[0] = u;
}

ConstExpr::ConstExpr(const Type *t, uint8_t *u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt8->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingUInt8->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        uint8Val[j] = u[j];
}

ConstExpr::ConstExpr(const Type *t, int16_t i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt16->GetAsConstType()));
    int16Val[0] = i;
}

ConstExpr::ConstExpr(const Type *t, int16_t *i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt16->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingInt16->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int16Val[j] = i[j];
}

ConstExpr::ConstExpr(const Type *t, uint16_t u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt16->GetAsConstType()));
    uint16Val[0] = u;
}

ConstExpr::ConstExpr(const Type *t, uint16_t *u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt16->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingUInt16->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        uint16Val[j] = u[j];
}

ConstExpr::ConstExpr(const Type *t, int32_t i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt32->GetAsConstType()));
    int32Val[0] = i;
}

ConstExpr::ConstExpr(const Type *t, int32_t *i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt32->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingInt32->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int32Val[j] = i[j];
}

ConstExpr::ConstExpr(const Type *t, uint32_t u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt32->GetAsConstType()) ||
                       (CastType<EnumType>(type) != nullptr && type->IsUniformType()));
    uint32Val[0] = u;
}

ConstExpr::ConstExpr(const Type *t, uint32_t *u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt32->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingUInt32->GetAsConstType()) ||
                       (CastType<EnumType>(type) != nullptr));
    for (int j = 0; j < Count(); ++j)
        uint32Val[j] = u[j];
}

ConstExpr::ConstExpr(const Type *t, llvm::APFloat f, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformFloat16->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::UniformFloat->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::UniformDouble->GetAsConstType()));
    fpVal.push_back(f);
}

ConstExpr::ConstExpr(const Type *t, std::vector<llvm::APFloat> const &f, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformFloat16->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingFloat16->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::UniformFloat->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingFloat->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::UniformDouble->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingDouble->GetAsConstType()));
    for (int j = 0; j < Count(); ++j) {
        fpVal.push_back(f[j]);
    }
}

ConstExpr::ConstExpr(const Type *t, int64_t i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt64->GetAsConstType()));
    int64Val[0] = i;
}

ConstExpr::ConstExpr(const Type *t, int64_t *i, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformInt64->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingInt64->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        int64Val[j] = i[j];
}

ConstExpr::ConstExpr(const Type *t, uint64_t u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt64->GetAsConstType()));
    uint64Val[0] = u;
}

ConstExpr::ConstExpr(const Type *t, uint64_t *u, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformUInt64->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingUInt64->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        uint64Val[j] = u[j];
}

ConstExpr::ConstExpr(const Type *t, bool b, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformBool->GetAsConstType()));
    boolVal[0] = b;
}

ConstExpr::ConstExpr(const Type *t, bool *b, SourcePos p) : Expr(p, ConstExprID) {
    type = t;
    type = type->GetAsConstType();
    AssertPos(pos, Type::Equal(type, AtomicType::UniformBool->GetAsConstType()) ||
                       Type::Equal(type, AtomicType::VaryingBool->GetAsConstType()));
    for (int j = 0; j < Count(); ++j)
        boolVal[j] = b[j];
}

ConstExpr::ConstExpr(const ConstExpr *old, SourcePos p) : Expr(p, ConstExprID) {
    type = old->type;

    AtomicType::BasicType basicType = getBasicType();

    switch (basicType) {
    case AtomicType::TYPE_BOOL:
        std::copy(old->boolVal, old->boolVal + Count(), boolVal);
        break;
    case AtomicType::TYPE_INT8:
        std::copy(old->int8Val, old->int8Val + Count(), int8Val);
        break;
    case AtomicType::TYPE_UINT8:
        std::copy(old->uint8Val, old->uint8Val + Count(), uint8Val);
        break;
    case AtomicType::TYPE_INT16:
        std::copy(old->int16Val, old->int16Val + Count(), int16Val);
        break;
    case AtomicType::TYPE_UINT16:
        std::copy(old->uint16Val, old->uint16Val + Count(), uint16Val);
        break;
    case AtomicType::TYPE_INT32:
        std::copy(old->int32Val, old->int32Val + Count(), int32Val);
        break;
    case AtomicType::TYPE_UINT32:
        std::copy(old->uint32Val, old->uint32Val + Count(), uint32Val);
        break;
    case AtomicType::TYPE_INT64:
        std::copy(old->int64Val, old->int64Val + Count(), int64Val);
        break;
    case AtomicType::TYPE_UINT64:
        std::copy(old->uint64Val, old->uint64Val + Count(), uint64Val);
        break;
    case AtomicType::TYPE_FLOAT16:
        fpVal = old->fpVal;
        break;
    case AtomicType::TYPE_FLOAT:
        fpVal = old->fpVal;
        break;
    case AtomicType::TYPE_DOUBLE:
        fpVal = old->fpVal;
        break;
    default:
        FATAL("unimplemented const type");
    }
}

AtomicType::BasicType ConstExpr::getBasicType() const {
    const AtomicType *at = CastType<AtomicType>(type);
    if (at != nullptr)
        return at->basicType;
    else {
        AssertPos(pos, CastType<EnumType>(type) != nullptr);
        return AtomicType::TYPE_UINT32;
    }
}

const Type *ConstExpr::GetType() const { return type; }

llvm::Value *ConstExpr::GetValue(FunctionEmitContext *ctx) const {
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
        return isVarying ? LLVMInt8Vector(int8Val) : LLVMInt8(int8Val[0]);
    case AtomicType::TYPE_UINT8:
        return isVarying ? LLVMUInt8Vector(uint8Val) : LLVMUInt8(uint8Val[0]);
    case AtomicType::TYPE_INT16:
        return isVarying ? LLVMInt16Vector(int16Val) : LLVMInt16(int16Val[0]);
    case AtomicType::TYPE_UINT16:
        return isVarying ? LLVMUInt16Vector(uint16Val) : LLVMUInt16(uint16Val[0]);
    case AtomicType::TYPE_INT32:
        return isVarying ? LLVMInt32Vector(int32Val) : LLVMInt32(int32Val[0]);
    case AtomicType::TYPE_UINT32:
        return isVarying ? LLVMUInt32Vector(uint32Val) : LLVMUInt32(uint32Val[0]);
    case AtomicType::TYPE_INT64:
        return isVarying ? LLVMInt64Vector(int64Val) : LLVMInt64(int64Val[0]);
    case AtomicType::TYPE_UINT64:
        return isVarying ? LLVMUInt64Vector(uint64Val) : LLVMUInt64(uint64Val[0]);
    case AtomicType::TYPE_FLOAT16:
        return isVarying ? LLVMFloat16Vector(fpVal) : LLVMFloat16(fpVal[0]);
    case AtomicType::TYPE_FLOAT:
        return isVarying ? LLVMFloatVector(fpVal) : LLVMFloat(fpVal[0]);
    case AtomicType::TYPE_DOUBLE:
        return isVarying ? LLVMDoubleVector(fpVal) : LLVMDouble(fpVal[0]);
    default:
        FATAL("unimplemented const type");
        return nullptr;
    }
}

bool ConstExpr::IsEqual(const ConstExpr *ce) const {
    if (ce == nullptr)
        return false;

    if (!Type::EqualIgnoringConst(type, ce->type))
        return false;

    for (int i = 0; i < Count(); ++i) {
        switch (getBasicType()) {
        case AtomicType::TYPE_BOOL:
            if (boolVal[i] != ce->boolVal[i])
                return false;
            break;
        case AtomicType::TYPE_INT8:
            if (int8Val[i] != ce->int8Val[i])
                return false;
            break;
        case AtomicType::TYPE_UINT8:
            if (uint8Val[i] != ce->uint8Val[i])
                return false;
            break;
        case AtomicType::TYPE_INT16:
            if (int16Val[i] != ce->int16Val[i])
                return false;
            break;
        case AtomicType::TYPE_UINT16:
            if (uint16Val[i] != ce->uint16Val[i])
                return false;
            break;
        case AtomicType::TYPE_INT32:
            if (int32Val[i] != ce->int32Val[i])
                return false;
            break;
        case AtomicType::TYPE_UINT32:
            if (uint32Val[i] != ce->uint32Val[i])
                return false;
            break;
        case AtomicType::TYPE_INT64:
            if (int64Val[i] != ce->int64Val[i])
                return false;
            break;
        case AtomicType::TYPE_UINT64:
            if (uint64Val[i] != ce->uint64Val[i])
                return false;
            break;
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
        case AtomicType::TYPE_DOUBLE:
            if (fpVal[i] != ce->fpVal[i])
                return false;
            break;
        default:
            FATAL("unimplemented const type");
            return false;
        }
    }
    return true;
}
/* Type conversion templates: take advantage of C++ function overloading
   rules to get the one we want to match. */

/* First the most general case, just use C++ type conversion if nothing
   else matches */
template <typename From, typename To> static inline void lConvertElement(From from, To *to) { *to = (To)from; }

/** When converting from bool types to numeric types, make sure the result
    is one or zero.
 */
template <typename To> static inline void lConvertElement(bool from, To *to) { *to = from ? (To)1 : (To)0; }

/** When converting numeric types to bool, compare to zero.  (Do we
    actually need this one??) */
template <typename From> static inline void lConvertElement(From from, bool *to) { *to = (from != 0); }

/** And bool -> bool is just assignment */
static inline void lConvertElement(bool from, bool *to) { *to = from; }

/** When converting from floating types to numeric types, get value from APFloat. */
template <typename To> static inline void lConvertElement(llvm::APFloat from, To *to) {
    const llvm::fltSemantics &FS = (LLVMTypes::DoubleType)->getFltSemantics();
    bool ignored;
    from.convert(FS, llvm::APFloat::rmNearestTiesToEven, &ignored);
    double val = from.convertToDouble();
    *to = (To)val;
}

/** When converting numeric types to floating types, create new APFloat. */
template <typename From>
static inline void lConvertElement(From from, std::vector<llvm::APFloat> &to, llvm::Type *type) {
    llvm::APFloat apf = lCreateAPFloat((double)from, type);
    to.push_back(apf);
}

/** floating types -> floating types requires conversion. */
static inline void lConvertElement(llvm::APFloat from, std::vector<llvm::APFloat> &to, llvm::Type *type) {
    const llvm::fltSemantics &FS = type->getFltSemantics();
    bool ignored;
    from.convert(FS, llvm::APFloat::rmNearestTiesToEven, &ignored);
    to.push_back(from);
}

/** floating types -> bool is a special case. */
static inline void lConvertElement(llvm::APFloat from, bool *to) { *to = from.isNonZero(); }

/** bool -> floating types is also a special case. */
static inline void lConvertElement(bool from, std::vector<llvm::APFloat> &to, llvm::Type *type) {
    llvm::APFloat apf = lCreateAPFloat(from ? (double)1 : (double)0, type);
    to.push_back(apf);
}

/** Type conversion utility function
 */
template <typename From, typename To> static void lConvert(const From *from, To *to, int count, bool forceVarying) {
    for (int i = 0; i < count; ++i)
        lConvertElement(from[i], &to[i]);

    if (forceVarying && count == 1)
        for (int i = 1; i < g->target->getVectorWidth(); ++i)
            to[i] = to[0];
}

template <typename From>
static void lConvert(const From *from, std::vector<llvm::APFloat> &to, llvm::Type *type, int count, bool forceVarying) {
    for (int i = 0; i < count; ++i)
        lConvertElement(from[i], to, type);

    if (forceVarying && count == 1)
        for (int i = 1; i < g->target->getVectorWidth(); ++i)
            to.push_back(to[0]);
}

int ConstExpr::GetValues(std::vector<llvm::APFloat> &fpt) const {
    AtomicType::BasicType bType = getBasicType();
    AssertPos(pos, (bType == AtomicType::TYPE_FLOAT16) || (bType == AtomicType::TYPE_FLOAT) ||
                       (bType == AtomicType::TYPE_DOUBLE));
    fpt = fpVal;
    return Count();
}

int ConstExpr::GetValues(std::vector<llvm::APFloat> &fpt, llvm::Type *type, bool forceVarying) const {
    switch (getBasicType()) {
    case AtomicType::TYPE_BOOL:
        lConvert(boolVal, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_INT8:
        lConvert(int8Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_UINT8:
        lConvert(uint8Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_INT16:
        lConvert(int16Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_UINT16:
        lConvert(uint16Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_INT32:
        lConvert(int32Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_UINT32:
        lConvert(uint32Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_INT64:
        lConvert(int64Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_UINT64:
        lConvert(uint64Val, fpt, type, Count(), forceVarying);
        break;
    case AtomicType::TYPE_FLOAT16:
    case AtomicType::TYPE_FLOAT:
    case AtomicType::TYPE_DOUBLE:
        lConvert(&fpVal[0], fpt, type, Count(), forceVarying);
        break;
    default:
        FATAL("unimplemented const type");
    }
    return Count();
}

#define CONVERT_SWITCH                                                                                                 \
    switch (getBasicType()) {                                                                                          \
    case AtomicType::TYPE_BOOL:                                                                                        \
        lConvert(boolVal, toPtr, Count(), forceVarying);                                                               \
        break;                                                                                                         \
    case AtomicType::TYPE_INT8:                                                                                        \
        lConvert(int8Val, toPtr, Count(), forceVarying);                                                               \
        break;                                                                                                         \
    case AtomicType::TYPE_UINT8:                                                                                       \
        lConvert(uint8Val, toPtr, Count(), forceVarying);                                                              \
        break;                                                                                                         \
    case AtomicType::TYPE_INT16:                                                                                       \
        lConvert(int16Val, toPtr, Count(), forceVarying);                                                              \
        break;                                                                                                         \
    case AtomicType::TYPE_UINT16:                                                                                      \
        lConvert(uint16Val, toPtr, Count(), forceVarying);                                                             \
        break;                                                                                                         \
    case AtomicType::TYPE_INT32:                                                                                       \
        lConvert(int32Val, toPtr, Count(), forceVarying);                                                              \
        break;                                                                                                         \
    case AtomicType::TYPE_UINT32:                                                                                      \
        lConvert(uint32Val, toPtr, Count(), forceVarying);                                                             \
        break;                                                                                                         \
    case AtomicType::TYPE_INT64:                                                                                       \
        lConvert(int64Val, toPtr, Count(), forceVarying);                                                              \
        break;                                                                                                         \
    case AtomicType::TYPE_UINT64:                                                                                      \
        lConvert(uint64Val, toPtr, Count(), forceVarying);                                                             \
        break;                                                                                                         \
    case AtomicType::TYPE_FLOAT16:                                                                                     \
    case AtomicType::TYPE_FLOAT:                                                                                       \
    case AtomicType::TYPE_DOUBLE:                                                                                      \
        lConvert(&fpVal[0], toPtr, Count(), forceVarying);                                                             \
        break;                                                                                                         \
    default:                                                                                                           \
        FATAL("unimplemented const type");                                                                             \
    }                                                                                                                  \
    return Count();

int ConstExpr::GetValues(bool *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(int8_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(uint8_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(int16_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(uint16_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(int32_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(uint32_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(int64_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::GetValues(uint64_t *toPtr, bool forceVarying) const { CONVERT_SWITCH; }

int ConstExpr::Count() const { return GetType()->IsVaryingType() ? g->target->getVectorWidth() : 1; }

static std::pair<llvm::Constant *, bool> lGetConstExprConstant(const Type *constType, const ConstExpr *cExpr,
                                                               bool isStorageType) {
    // Caller shouldn't be trying to stuff a varying value here into a
    // constant type.
    SourcePos pos = cExpr->pos;
    bool isNotValidForMultiTargetGlobal = false;
    if (constType->IsUniformType())
        AssertPos(pos, cExpr->Count() == 1);

    constType = constType->GetAsNonConstType();
    if (Type::Equal(constType, AtomicType::UniformBool) || Type::Equal(constType, AtomicType::VaryingBool)) {
        bool bv[ISPC_MAX_NVEC];
        cExpr->GetValues(bv, constType->IsVaryingType());
        if (constType->IsUniformType()) {
            if (isStorageType)
                return std::pair<llvm::Constant *, bool>(bv[0] ? LLVMTrueInStorage : LLVMFalseInStorage,
                                                         isNotValidForMultiTargetGlobal);
            else
                return std::pair<llvm::Constant *, bool>(bv[0] ? LLVMTrue : LLVMFalse, isNotValidForMultiTargetGlobal);
        } else {
            if (isStorageType)
                return std::pair<llvm::Constant *, bool>(LLVMBoolVectorInStorage(bv), isNotValidForMultiTargetGlobal);
            else
                return std::pair<llvm::Constant *, bool>(LLVMBoolVector(bv), isNotValidForMultiTargetGlobal);
        }
    } else if (Type::Equal(constType, AtomicType::UniformInt8) || Type::Equal(constType, AtomicType::VaryingInt8)) {
        int8_t iv[ISPC_MAX_NVEC];
        cExpr->GetValues(iv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMInt8(iv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMInt8Vector(iv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformUInt8) || Type::Equal(constType, AtomicType::VaryingUInt8)) {
        uint8_t uiv[ISPC_MAX_NVEC];
        cExpr->GetValues(uiv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMUInt8(uiv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMUInt8Vector(uiv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformInt16) || Type::Equal(constType, AtomicType::VaryingInt16)) {
        int16_t iv[ISPC_MAX_NVEC];
        cExpr->GetValues(iv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMInt16(iv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMInt16Vector(iv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformUInt16) || Type::Equal(constType, AtomicType::VaryingUInt16)) {
        uint16_t uiv[ISPC_MAX_NVEC];
        cExpr->GetValues(uiv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMUInt16(uiv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMUInt16Vector(uiv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformInt32) || Type::Equal(constType, AtomicType::VaryingInt32)) {
        int32_t iv[ISPC_MAX_NVEC];
        cExpr->GetValues(iv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMInt32(iv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMInt32Vector(iv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformUInt32) || Type::Equal(constType, AtomicType::VaryingUInt32) ||
               CastType<EnumType>(constType) != nullptr) {
        uint32_t uiv[ISPC_MAX_NVEC];
        cExpr->GetValues(uiv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMUInt32(uiv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMUInt32Vector(uiv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformInt64) || Type::Equal(constType, AtomicType::VaryingInt64)) {
        int64_t iv[ISPC_MAX_NVEC];
        cExpr->GetValues(iv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMInt64(iv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMInt64Vector(iv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformUInt64) || Type::Equal(constType, AtomicType::VaryingUInt64)) {
        uint64_t uiv[ISPC_MAX_NVEC];
        cExpr->GetValues(uiv, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMUInt64(uiv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMUInt64Vector(uiv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformFloat16) ||
               Type::Equal(constType, AtomicType::VaryingFloat16)) {
        std::vector<llvm::APFloat> f16v;
        cExpr->GetValues(f16v, LLVMTypes::Float16Type, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMFloat16(f16v[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMFloat16Vector(f16v), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformFloat) || Type::Equal(constType, AtomicType::VaryingFloat)) {
        std::vector<llvm::APFloat> fv;
        cExpr->GetValues(fv, LLVMTypes::FloatType, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMFloat(fv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMFloatVector(fv), isNotValidForMultiTargetGlobal);
    } else if (Type::Equal(constType, AtomicType::UniformDouble) || Type::Equal(constType, AtomicType::VaryingDouble)) {
        std::vector<llvm::APFloat> dv;
        cExpr->GetValues(dv, LLVMTypes::DoubleType, constType->IsVaryingType());
        if (constType->IsUniformType())
            return std::pair<llvm::Constant *, bool>(LLVMDouble(dv[0]), isNotValidForMultiTargetGlobal);
        else
            return std::pair<llvm::Constant *, bool>(LLVMDoubleVector(dv), isNotValidForMultiTargetGlobal);
    } else if (CastType<PointerType>(constType) != nullptr) {
        // The only time we should get here is if we have an integer '0'
        // constant that should be turned into a nullptr pointer of the
        // appropriate type.
        llvm::Type *llvmType = constType->LLVMType(g->ctx);
        if (llvmType == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return std::pair<llvm::Constant *, bool>(nullptr, false);
        }

        int64_t iv[ISPC_MAX_NVEC];
        cExpr->GetValues(iv, constType->IsVaryingType());
        for (int i = 0; i < cExpr->Count(); ++i)
            if (iv[i] != 0)
                // We'll issue an error about this later--trying to assign
                // a constant int to a pointer, without a typecast.
                return std::pair<llvm::Constant *, bool>(nullptr, false);

        return std::pair<llvm::Constant *, bool>(llvm::Constant::getNullValue(llvmType),
                                                 isNotValidForMultiTargetGlobal);
    } else {
        Debug(pos, "Unable to handle type \"%s\" in ConstExpr::GetConstant().", constType->GetString().c_str());
        return std::pair<llvm::Constant *, bool>(nullptr, isNotValidForMultiTargetGlobal);
    }
}

std::pair<llvm::Constant *, bool> ConstExpr::GetStorageConstant(const Type *constType) const {
    return lGetConstExprConstant(constType, this, true);
}
std::pair<llvm::Constant *, bool> ConstExpr::GetConstant(const Type *constType) const {
    return lGetConstExprConstant(constType, this, false);
}

Expr *ConstExpr::Optimize() { return this; }

Expr *ConstExpr::TypeCheck() { return this; }

int ConstExpr::EstimateCost() const { return 0; }

ConstExpr *ConstExpr::Instantiate(TemplateInstantiation &templInst) const { return new ConstExpr(this, pos); }

std::string ConstExpr::GetValuesAsStr(const std::string &separator) const {
    std::stringstream result;
    for (int i = 0; i < Count(); ++i) {
        if (i != 0) {
            result << separator;
        }
        switch (getBasicType()) {
        case AtomicType::TYPE_BOOL:
            result << (boolVal[i] ? "true" : "false");
            break;
        case AtomicType::TYPE_INT8:
            result << static_cast<int>(int8Val[i]);
            break;
        case AtomicType::TYPE_UINT8:
            result << static_cast<unsigned int>(uint8Val[i]);
            break;
        case AtomicType::TYPE_INT16:
            result << static_cast<int>(int16Val[i]);
            break;
        case AtomicType::TYPE_UINT16:
            result << static_cast<unsigned int>(uint16Val[i]);
            break;
        case AtomicType::TYPE_INT32:
            result << int32Val[i];
            break;
        case AtomicType::TYPE_UINT32:
            result << uint32Val[i];
            break;
        case AtomicType::TYPE_INT64:
            result << int64Val[i];
            break;
        case AtomicType::TYPE_UINT64:
            result << uint64Val[i];
            break;
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
            result << std::to_string(fpVal[i].convertToFloat());
            break;
        case AtomicType::TYPE_DOUBLE:
            result << std::to_string(fpVal[i].convertToDouble());
            break;
        default:
            FATAL("unimplemented const type");
        }
    }
    return result.str();
}

void ConstExpr::Print(Indent &indent) const {
    indent.Print("ConstExpr", pos);

    printf("[%s] (", GetType()->GetString().c_str());
    printf("%s", GetValuesAsStr((char *)", ").c_str());
    printf(")\n");

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// TypeCastExpr

TypeCastExpr::TypeCastExpr(const Type *t, Expr *e, SourcePos p) : Expr(p, TypeCastExprID) {
    type = t;
    expr = e;
}

/** Handle all the grungy details of type conversion between atomic types and uniform vector types.
    Given an input value in exprVal of type fromType, convert it to the
    llvm::Value with type toType.
 */
static llvm::Value *lTypeConvAtomicOrUniformVector(FunctionEmitContext *ctx, llvm::Value *exprVal, const Type *toType,
                                                   const Type *fromType, SourcePos pos) {
    AtomicType::BasicType basicToType;
    AtomicType::BasicType basicFromType;
    if (toType->IsVectorType() && toType->IsUniformType() && fromType->IsVectorType() && fromType->IsUniformType()) {
        const VectorType *vToType = CastType<VectorType>(toType);
        const VectorType *vFromType = CastType<VectorType>(fromType);
        basicToType = vToType->GetElementType()->basicType;
        basicFromType = vFromType->GetElementType()->basicType;
    } else if (toType->IsAtomicType() && fromType->IsAtomicType()) {
        const AtomicType *aToType = CastType<AtomicType>(toType);
        const AtomicType *aFromType = CastType<AtomicType>(fromType);
        basicToType = aToType->basicType;
        basicFromType = aFromType->basicType;
    } else {
        FATAL("Unexpected input type in lTypeConvAtomicOrUniformVector");
    }
    llvm::Value *cast = nullptr;

    std::string opName = exprVal->getName().str();
    switch (basicToType) {
    case AtomicType::TYPE_BOOL:
        opName += "_to_bool";
        break;
    case AtomicType::TYPE_INT8:
        opName += "_to_int8";
        break;
    case AtomicType::TYPE_UINT8:
        opName += "_to_uint8";
        break;
    case AtomicType::TYPE_INT16:
        opName += "_to_int16";
        break;
    case AtomicType::TYPE_UINT16:
        opName += "_to_uint16";
        break;
    case AtomicType::TYPE_INT32:
        opName += "_to_int32";
        break;
    case AtomicType::TYPE_UINT32:
        opName += "_to_uint32";
        break;
    case AtomicType::TYPE_INT64:
        opName += "_to_int64";
        break;
    case AtomicType::TYPE_UINT64:
        opName += "_to_uint64";
        break;
    case AtomicType::TYPE_FLOAT16:
        opName += "_to_float16";
        break;
    case AtomicType::TYPE_FLOAT:
        opName += "_to_float";
        break;
    case AtomicType::TYPE_DOUBLE:
        opName += "_to_double";
        break;
    default:
        FATAL("Unimplemented");
    }
    const char *cOpName = opName.c_str();

    llvm::Type *targetType = fromType->IsUniformType() ? toType->GetAsUniformType()->LLVMType(g->ctx)
                                                       : toType->GetAsVaryingType()->LLVMType(g->ctx);

    switch (basicToType) {
    case AtomicType::TYPE_FLOAT16: {
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                // If we have a bool vector of non-i1 elements, first
                // truncate down to a single bit.
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
            // And then do an unisgned int->float cast
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_INT64:
            cast = ctx->CastInst(llvm::Instruction::SIToFP, // signed int to float16
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT8:
        case AtomicType::TYPE_UINT16:
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float16
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT32:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat16)) {
                PerformanceWarning(pos, "Conversion from uint32 to float16 is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float16
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT64:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat16)) {
                PerformanceWarning(pos, "Conversion from uint64 to float16 is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float16
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT16:
            // No-op cast.
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->FPCastInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->FPCastInst(exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_FLOAT: {
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                // If we have a bool vector of non-i1 elements, first
                // truncate down to a single bit.
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT32:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from uint32 to float is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT64:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from uint64 to float is slow. Use \"int64\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int to float
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT16:
            cast = ctx->FPCastInst(exprVal, targetType, cOpName);
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
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                // truncate bool vector values to i1s if necessary.
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT32:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from uint32 to double is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_UINT64:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from uint64 to double is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::UIToFP, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT16:
            cast = ctx->FPCastInst(exprVal, targetType, cOpName);
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
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
        case AtomicType::TYPE_FLOAT16:
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
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic()) {
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
            }
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
        case AtomicType::TYPE_FLOAT16:
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT16: {
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
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
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32:
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT16:
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT32: {
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
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
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = ctx->TruncInst(exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT16:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat16)) {
                PerformanceWarning(pos, "Conversion from float16 to uint32 is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from float to uint32 is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from double to uint32 is slow. Use \"int32\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // unsigned int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_INT64: {
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
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
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic())
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
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
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64:
            cast = exprVal;
            break;
        case AtomicType::TYPE_FLOAT16:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat16)) {
                PerformanceWarning(pos, "Conversion from float16 to uint64 is slow. Use \"int64\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_FLOAT:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from float to uint64 is slow. Use \"int64\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        case AtomicType::TYPE_DOUBLE:
            if (fromType->IsVaryingAtomicOrUniformVectorType() &&
                g->target->shouldWarn(PerfWarningType::CVTUIntFloat)) {
                PerformanceWarning(pos, "Conversion from double to uint64 is slow. Use \"int64\" if possible");
            }
            cast = ctx->CastInst(llvm::Instruction::FPToUI, // signed int
                                 exprVal, targetType, cOpName);
            break;
        default:
            FATAL("unimplemented");
        }
        break;
    }
    case AtomicType::TYPE_BOOL: {
        switch (basicFromType) {
        case AtomicType::TYPE_BOOL:
            if (fromType->IsVaryingAtomic()) {
                // truncate bool vector values to i1s if necessary.
                exprVal = ctx->SwitchBoolToMaskType(exprVal, LLVMTypes::Int1VectorType, cOpName);
            }
            cast = exprVal;
            break;
        case AtomicType::TYPE_INT8:
        case AtomicType::TYPE_UINT8: {
            llvm::Value *zero = LLVMIntAsType(0, fromType->LLVMType(g->ctx));
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_INT16:
        case AtomicType::TYPE_UINT16: {
            llvm::Value *zero = LLVMIntAsType(0, fromType->LLVMType(g->ctx));
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_INT32:
        case AtomicType::TYPE_UINT32: {
            llvm::Value *zero = LLVMIntAsType(0, fromType->LLVMType(g->ctx));
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_INT64:
        case AtomicType::TYPE_UINT64: {
            llvm::Value *zero = LLVMIntAsType(0, fromType->LLVMType(g->ctx));
            cast = ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_FLOAT16: {
            llvm::Value *zero = LLVMFPZeroAsType(fromType->LLVMType(g->ctx));
            cast = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_ONE, exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_FLOAT: {
            llvm::Value *zero = LLVMFPZeroAsType(fromType->LLVMType(g->ctx));
            cast = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_ONE, exprVal, zero, cOpName);
            break;
        }
        case AtomicType::TYPE_DOUBLE: {
            llvm::Value *zero = LLVMFPZeroAsType(fromType->LLVMType(g->ctx));
            cast = ctx->CmpInst(llvm::Instruction::FCmp, llvm::CmpInst::FCMP_ONE, exprVal, zero, cOpName);
            break;
        }
        default:
            FATAL("unimplemented");
        }

        if (fromType->IsUniformType()) {
            if (toType->IsAtomicType() && toType->IsVaryingType() &&
                LLVMTypes::BoolVectorType != LLVMTypes::Int1VectorType) {
                // extend out to an bool as an i8/i16/i32 from the i1 here.
                // Then we'll turn that into a vector below, the way it
                // does for everyone else...
                Assert(cast);
                cast = ctx->SwitchBoolToMaskType(cast, LLVMTypes::BoolVectorType->getElementType(),
                                                 llvm::Twine(cast->getName()) + "to_i_bool");
            }
        } else {
            // fromType->IsVaryingType())
            cast = ctx->I1VecToBoolVec(cast);
        }
        break;
    }
    default:
        FATAL("unimplemented");
    }

    // If we also want to go from uniform to varying, replicate out the
    // value across the vector elements..
    if (toType->IsVaryingType() && toType->IsAtomicType() && fromType->IsUniformType())
        return ctx->SmearUniform(cast);
    else
        return cast;
}

// FIXME: fold this into the FunctionEmitContext::SmearUniform() method?

/** Converts the given value of the given type to be the varying
    equivalent, returning the resulting value.
 */
static llvm::Value *lUniformValueToVarying(FunctionEmitContext *ctx, llvm::Value *value, const Type *type,
                                           SourcePos pos) {
    // nothing to do if it's already varying
    if (type->IsVaryingType())
        return value;

    // for structs/arrays/vectors, just recursively make their elements
    // varying (if needed) and populate the return value.
    const CollectionType *collectionType = CastType<CollectionType>(type);
    if (collectionType != nullptr) {
        llvm::Type *llvmType = type->GetAsVaryingType()->LLVMStorageType(g->ctx);
        llvm::Value *retValue = llvm::UndefValue::get(llvmType);

        const StructType *structType = CastType<StructType>(type->GetAsVaryingType());

        for (int i = 0; i < collectionType->GetElementCount(); ++i) {
            llvm::Value *v = ctx->ExtractInst(value, i, "get_element");
            // If struct has "bound uniform" member, we don't need to cast it to varying
            if (!(structType != nullptr && structType->GetElementType(i)->IsUniformType())) {
                const Type *elemType = collectionType->GetElementType(i);
                // If member is a uniform bool, it needs to be truncated to i1 since
                // uniform  bool in IR is i1 and i8 in struct
                // Consider switching to just a broadcast for bool
                if ((elemType->IsBoolType()) && (CastType<AtomicType>(elemType) != nullptr)) {
                    v = ctx->TruncInst(v, LLVMTypes::BoolType);
                }
                v = lUniformValueToVarying(ctx, v, elemType, pos);
                // If the extracted element if bool and varying needs to be
                // converted back to i8 vector to insert into varying struct.
                if ((elemType->IsBoolType()) && (CastType<AtomicType>(elemType) != nullptr)) {
                    v = ctx->SwitchBoolToStorageType(v, LLVMTypes::BoolVectorStorageType);
                }
            }
            retValue = ctx->InsertInst(retValue, v, i, "set_element");
        }
        return retValue;
    }

    // Otherwise we must have a uniform atomic or pointer type, so smear
    // its value across the vector lanes.
    if (CastType<AtomicType>(type) && CastType<AtomicType>(type->GetAsVaryingType())) {
        return lTypeConvAtomicOrUniformVector(ctx, value, CastType<AtomicType>(type->GetAsVaryingType()),
                                              CastType<AtomicType>(type), pos);
    }

    Assert(CastType<PointerType>(type) != nullptr);
    return ctx->SmearUniform(value);
}

bool TypeCastExpr::HasAmbiguousVariability(std::vector<const Expr *> &warn) const {

    if (expr == nullptr)
        return false;

    const Type *toType = type, *fromType = expr->GetType();
    if (toType == nullptr || fromType == nullptr)
        return false;

    if (toType->HasUnboundVariability() && fromType->IsUniformType()) {
        warn.push_back(this);
        return true;
    }

    return false;
}

void TypeCastExpr::PrintAmbiguousVariability() const {
    const Type *exprType = expr->GetType();
    Assert(exprType);
    Warning(pos,
            "Typecasting to type \"%s\" (variability not specified) "
            "from \"uniform\" type \"%s\" results in \"uniform\" variability.\n"
            "In the context of function argument it may lead to unexpected behavior. "
            "Casting to \"%s\" is recommended.",
            (type->GetString()).c_str(), (exprType->GetString()).c_str(),
            (type->GetAsUniformType()->GetString()).c_str());
}

llvm::Value *TypeCastExpr::GetValue(FunctionEmitContext *ctx) const {
    if (!expr)
        return nullptr;

    ctx->SetDebugPos(pos);
    const Type *toType = GetType(), *fromType = expr->GetType();
    if (toType == nullptr || fromType == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (toType->IsVoidType()) {
        // emit the code for the expression in case it has side-effects but
        // then we're done.
        (void)expr->GetValue(ctx);
        return nullptr;
    }

    const PointerType *fromPointerType = CastType<PointerType>(fromType);
    const PointerType *toPointerType = CastType<PointerType>(toType);
    const ArrayType *toArrayType = CastType<ArrayType>(toType);
    const ArrayType *fromArrayType = CastType<ArrayType>(fromType);
    if (fromPointerType != nullptr) {
        if (toArrayType != nullptr) {
            return expr->GetValue(ctx);
        } else if (toPointerType != nullptr) {
            llvm::Value *value = expr->GetValue(ctx);
            if (value == nullptr)
                return nullptr;

            if (fromPointerType->IsSlice() == false && toPointerType->IsSlice() == true) {
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
            } else {
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
                } else {
                    // Otherwise we just bitcast it to an int and smear it
                    // out to a vector
                    value = ctx->PtrToIntInst(value);
                    return ctx->SmearUniform(value);
                }
            }
        } else {
            AssertPos(pos, CastType<AtomicType>(toType) != nullptr);
            if (toType->IsBoolType()) {
                // convert pointer to bool
                llvm::Type *lfu = fromType->GetAsUniformType()->LLVMType(g->ctx);
                llvm::PointerType *llvmFromUnifType = llvm::dyn_cast<llvm::PointerType>(lfu);

                llvm::Value *nullPtrValue = llvm::ConstantPointerNull::get(llvmFromUnifType);
                if (fromType->IsVaryingType())
                    nullPtrValue = ctx->SmearUniform(nullPtrValue);

                llvm::Value *exprVal = expr->GetValue(ctx);
                llvm::Value *cmp =
                    ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, exprVal, nullPtrValue, "ptr_ne_null");

                if (toType->IsVaryingType()) {
                    if (fromType->IsUniformType())
                        cmp = ctx->SmearUniform(cmp);
                    cmp = ctx->I1VecToBoolVec(cmp);
                }

                return cmp;
            } else {
                // ptr -> int
                llvm::Value *value = expr->GetValue(ctx);
                if (value == nullptr)
                    return nullptr;

                if (toType->IsVaryingType() && fromType->IsUniformType())
                    value = ctx->SmearUniform(value);

                llvm::Type *llvmToType = toType->LLVMType(g->ctx);
                if (llvmToType == nullptr)
                    return nullptr;
                return ctx->PtrToIntInst(value, llvmToType, "ptr_typecast");
            }
        }
    }

    if (Type::EqualIgnoringConst(toType, fromType))
        // There's nothing to do, just return the value.  (LLVM's type
        // system doesn't worry about constiness.)
        return expr->GetValue(ctx);

    if (fromArrayType != nullptr && toPointerType != nullptr) {
        // implicit array to pointer to first element
        Expr *arrayAsPtr = lArrayToPointer(expr);
        const Type *arrayType = arrayAsPtr->GetType();
        Assert(arrayType);
        if (Type::EqualIgnoringConst(arrayType, toPointerType) == false) {
            AssertPos(pos, PointerType::IsVoidPointer(toPointerType) ||
                               Type::EqualIgnoringConst(arrayType->GetAsVaryingType(), toPointerType) == true);
            arrayAsPtr = new TypeCastExpr(toPointerType, arrayAsPtr, pos);
            arrayAsPtr = ::TypeCheck(arrayAsPtr);
            AssertPos(pos, arrayAsPtr != nullptr);
            arrayAsPtr = ::Optimize(arrayAsPtr);
            AssertPos(pos, arrayAsPtr != nullptr);
            arrayType = arrayAsPtr->GetType();
        }
        Assert(arrayType);
        AssertPos(pos, Type::EqualIgnoringConst(arrayType, toPointerType));
        return arrayAsPtr->GetValue(ctx);
    }

    // This also should be caught during typechecking
    AssertPos(pos, !(toType->IsUniformType() && fromType->IsVaryingType()));

    if (toArrayType != nullptr && fromArrayType != nullptr) {
        // cast array pointer from [n x foo] to [0 x foo] if needed to be able
        // to pass to a function that takes an unsized array as a parameter
        if (toArrayType->GetElementCount() != 0 && (toArrayType->GetElementCount() != fromArrayType->GetElementCount()))
            Warning(pos, "Type-converting array of length %d to length %d", fromArrayType->GetElementCount(),
                    toArrayType->GetElementCount());
        AssertPos(pos, Type::EqualIgnoringConst(toArrayType->GetBaseType(), fromArrayType->GetBaseType()));
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
            if (toArray->GetElementCount() != 0 && (toArray->GetElementCount() != fromArray->GetElementCount()))
                Warning(pos, "Type-converting array of length %d to length %d", fromArray->GetElementCount(),
                        toArray->GetElementCount());
            AssertPos(pos, Type::EqualIgnoringConst(toArray->GetBaseType(), fromArray->GetBaseType()));
            llvm::Value *v = expr->GetValue(ctx);
            llvm::Type *ptype = toType->LLVMType(g->ctx);
            return ctx->BitCastInst(v, ptype); //, "array_cast_0size");
        }

        // Just bitcast it.  See Issue #721
        llvm::Value *value = expr->GetValue(ctx);
        return ctx->BitCastInst(value, toType->LLVMType(g->ctx), "refcast");
    }

    const StructType *toStruct = CastType<StructType>(toType);
    const StructType *fromStruct = CastType<StructType>(fromType);
    if (toStruct && fromStruct) {
        // The only legal type conversions for structs are to go from a
        // uniform to a varying instance of the same struct type.
        AssertPos(pos, toStruct->IsVaryingType() && fromStruct->IsUniformType() &&
                           Type::EqualIgnoringConst(toStruct, fromStruct->GetAsVaryingType()));

        llvm::Value *origValue = expr->GetValue(ctx);
        if (!origValue)
            return nullptr;
        return lUniformValueToVarying(ctx, origValue, fromType, pos);
    }

    const VectorType *toVector = CastType<VectorType>(toType);
    const VectorType *fromVector = CastType<VectorType>(fromType);
    if (toVector && fromVector) {
        // this should be caught during typechecking
        AssertPos(pos, toVector->GetElementCount() == fromVector->GetElementCount());

        llvm::Value *exprVal = expr->GetValue(ctx);
        if (!exprVal)
            return nullptr;

        // Check if we have two uniform vectors
        if (fromVector->IsUniformType() && toVector->IsUniformType()) {
            llvm::Value *conv = lTypeConvAtomicOrUniformVector(ctx, exprVal, toVector, fromVector, pos);
            if (!conv)
                return nullptr;
            if ((toVector->GetElementType()->IsBoolType())) {
                conv = ctx->SwitchBoolToStorageType(conv, toVector->LLVMStorageType(g->ctx));
            }
            return conv;
        } else {
            // Emit instructions to do type conversion of each of the elements
            // of the vector.
            llvm::Value *cast = llvm::UndefValue::get(toType->LLVMStorageType(g->ctx));
            for (int i = 0; i < toVector->GetElementCount(); ++i) {
                llvm::Value *ei = ctx->ExtractInst(exprVal, i);
                llvm::Value *conv = lTypeConvAtomicOrUniformVector(ctx, ei, toVector->GetElementType(),
                                                                   fromVector->GetElementType(), pos);
                if (!conv)
                    return nullptr;
                if ((toVector->GetElementType()->IsBoolType()) &&
                    (CastType<AtomicType>(toVector->GetElementType()) != nullptr)) {
                    conv = ctx->SwitchBoolToStorageType(conv, toVector->GetElementType()->LLVMStorageType(g->ctx));
                }

                cast = ctx->InsertInst(cast, conv, i);
            }
            return cast;
        }
    }

    llvm::Value *exprVal = expr->GetValue(ctx);
    if (!exprVal)
        return nullptr;

    const EnumType *fromEnum = CastType<EnumType>(fromType);
    const EnumType *toEnum = CastType<EnumType>(toType);
    if (fromEnum)
        // treat it as an uint32 type for the below and all will be good.
        fromType = fromEnum->IsUniformType() ? AtomicType::UniformUInt32 : AtomicType::VaryingUInt32;
    if (toEnum)
        // treat it as an uint32 type for the below and all will be good.
        toType = toEnum->IsUniformType() ? AtomicType::UniformUInt32 : AtomicType::VaryingUInt32;

    const AtomicType *fromAtomic = CastType<AtomicType>(fromType);
    // at this point, coming from an atomic type is all that's left...
    AssertPos(pos, fromAtomic != nullptr);

    if (toVector) {
        // scalar -> short vector conversion
        llvm::Value *conv = lTypeConvAtomicOrUniformVector(ctx, exprVal, toVector->GetElementType(), fromAtomic, pos);
        if (!conv)
            return nullptr;

        llvm::Value *cast = nullptr;
        llvm::Type *toTypeLLVM = toType->LLVMStorageType(g->ctx);
        if (llvm::isa<llvm::VectorType>(toTypeLLVM)) {
            // Example uniform float => uniform float<3>
            cast = ctx->BroadcastValue(conv, toTypeLLVM);
        } else if (llvm::isa<llvm::ArrayType>(toTypeLLVM)) {
            // Example varying float => varying float<3>
            cast = llvm::UndefValue::get(toType->LLVMStorageType(g->ctx));
            for (int i = 0; i < toVector->GetElementCount(); ++i) {
                if ((toVector->GetElementType()->IsBoolType()) &&
                    (CastType<AtomicType>(toVector->GetElementType()) != nullptr)) {
                    conv = ctx->SwitchBoolToStorageType(conv, toVector->GetElementType()->LLVMStorageType(g->ctx));
                }
                // Here's InsertInst produces InsertValueInst.
                cast = ctx->InsertInst(cast, conv, i);
            }
        } else {
            FATAL("TypeCastExpr::GetValue: problem with cast");
        }

        return cast;
    } else if (toPointerType != nullptr) {
        // int -> ptr
        if (toType->IsVaryingType() && fromType->IsUniformType())
            exprVal = ctx->SmearUniform(exprVal);

        llvm::Type *llvmToType = toType->LLVMType(g->ctx);
        if (llvmToType == nullptr)
            return nullptr;

        return ctx->IntToPtrInst(exprVal, llvmToType, "int_to_ptr");
    } else {
        const AtomicType *toAtomic = CastType<AtomicType>(toType);
        // typechecking should ensure this is the case
        AssertPos(pos, toAtomic != nullptr);

        return lTypeConvAtomicOrUniformVector(ctx, exprVal, toAtomic, fromAtomic, pos);
    }
}

llvm::Value *TypeCastExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (GetLValueType() != nullptr) {
        return GetValue(ctx);
    } else {
        return nullptr;
    }
}

const Type *TypeCastExpr::GetType() const {
    // Here we try to resolve situation where (base_type) can be treated as
    // (uniform base_type) of (varying base_type). This is a part of function
    // TypeCastExpr::TypeCheck. After implementation of operators we
    // have to have this functionality here.
    if (expr == nullptr)
        return nullptr;
    const Type *toType = type, *fromType = expr->GetType();
    if (toType == nullptr || fromType == nullptr)
        return nullptr;

    if (toType->IsDependentType()) {
        return toType;
    }

    if (toType->HasUnboundVariability()) {
        if (fromType->IsUniformType()) {
            toType = type->ResolveUnboundVariability(Variability::Uniform);
        } else {
            toType = type->ResolveUnboundVariability(Variability::Varying);
        }
    }
    AssertPos(pos, toType->HasUnboundVariability() == false);
    return toType;
}

const Type *TypeCastExpr::GetLValueType() const {
    AssertPos(pos, type->HasUnboundVariability() == false);
    if (CastType<PointerType>(GetType()) != nullptr) {
        return type;
    } else {
        return nullptr;
    }
}

static const Type *lDeconstifyType(const Type *t) {
    const PointerType *pt = CastType<PointerType>(t);
    if (pt != nullptr)
        return new PointerType(lDeconstifyType(pt->GetBaseType()), pt->GetVariability(), false);
    else
        return t->GetAsNonConstType();
}

Expr *TypeCastExpr::TypeCheck() {
    if (expr == nullptr)
        return nullptr;

    const Type *toType = type, *fromType = expr->GetType();
    if (toType == nullptr || fromType == nullptr)
        return nullptr;

    if (toType->IsDependentType() || fromType->IsDependentType()) {
        return this;
    }

    if (toType->HasUnboundVariability() && fromType->IsUniformType()) {
        TypeCastExpr *tce = new TypeCastExpr(toType->GetAsUniformType(), expr, pos);
        return ::TypeCheck(tce);
    }
    type = toType = type->ResolveUnboundVariability(Variability::Varying);

    fromType = lDeconstifyType(fromType);
    toType = lDeconstifyType(toType);

    // Anything can be cast to void...
    if (toType->IsVoidType())
        return this;

    if (fromType->IsVoidType() || (fromType->IsVaryingType() && toType->IsUniformType())) {
        Error(pos, "Can't type cast from type \"%s\" to type \"%s\"", fromType->GetString().c_str(),
              toType->GetString().c_str());
        return nullptr;
    }

    // First some special cases that we allow only with an explicit type cast
    const PointerType *fromPtr = CastType<PointerType>(fromType);
    const PointerType *toPtr = CastType<PointerType>(toType);
    if (fromPtr != nullptr && toPtr != nullptr)
        // allow explicit typecasts between any two different pointer types
        return this;

    const ReferenceType *fromRef = CastType<ReferenceType>(fromType);
    const ReferenceType *toRef = CastType<ReferenceType>(toType);
    if (fromRef != nullptr && toRef != nullptr) {
        // allow explicit typecasts between any two different reference types
        // Issues #721
        return this;
    }

    const AtomicType *fromAtomic = CastType<AtomicType>(fromType);
    const AtomicType *toAtomic = CastType<AtomicType>(toType);
    const EnumType *fromEnum = CastType<EnumType>(fromType);
    const EnumType *toEnum = CastType<EnumType>(toType);
    if ((fromAtomic || fromEnum) && (toAtomic || toEnum))
        // Allow explicit casts between all of these
        return this;

    // ptr -> int type casts
    if (fromPtr != nullptr && toAtomic != nullptr && toAtomic->IsIntType()) {
        bool safeCast =
            (toAtomic->basicType == AtomicType::TYPE_INT64 || toAtomic->basicType == AtomicType::TYPE_UINT64);
        if (g->target->is32Bit())
            safeCast |=
                (toAtomic->basicType == AtomicType::TYPE_INT32 || toAtomic->basicType == AtomicType::TYPE_UINT32);
        if (safeCast == false)
            Warning(pos,
                    "Pointer type cast of type \"%s\" to integer type "
                    "\"%s\" may lose information.",
                    fromType->GetString().c_str(), toType->GetString().c_str());
        return this;
    }

    // int -> ptr
    if (fromAtomic != nullptr && fromAtomic->IsIntType() && toPtr != nullptr)
        return this;

    // And otherwise see if it's one of the conversions allowed to happen
    // implicitly.
    Expr *e = TypeConvertExpr(expr, toType, "type cast expression");
    if (e == nullptr)
        return nullptr;
    else
        return e;
}

Expr *TypeCastExpr::Optimize() {
    ConstExpr *constExpr = llvm::dyn_cast<ConstExpr>(expr);
    if (constExpr == nullptr)
        // We can't do anything if this isn't a const expr
        return this;

    const Type *toType = GetType();
    const AtomicType *toAtomic = CastType<AtomicType>(toType);
    const EnumType *toEnum = CastType<EnumType>(toType);
    // If we're not casting to an atomic or enum type, we can't do anything
    // here, since ConstExprs can only represent those two types.  (So
    // e.g. we're casting from an int to an int<4>.)
    if (toAtomic == nullptr && toEnum == nullptr)
        return this;

    bool forceVarying = toType->IsVaryingType();

    // All of the type conversion smarts we need is already in the
    // ConstExpr::GetValues(), etc., methods, so we just need to call the
    // appropriate one for the type that this cast is converting to.
    AtomicType::BasicType basicType = toAtomic ? toAtomic->basicType : AtomicType::TYPE_UINT32;
    switch (basicType) {
    case AtomicType::TYPE_BOOL: {
        bool bv[ISPC_MAX_NVEC];
        constExpr->GetValues(bv, forceVarying);
        return new ConstExpr(toType, bv, pos);
    }
    case AtomicType::TYPE_INT8: {
        int8_t iv[ISPC_MAX_NVEC];
        constExpr->GetValues(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT8: {
        uint8_t uv[ISPC_MAX_NVEC];
        constExpr->GetValues(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_INT16: {
        int16_t iv[ISPC_MAX_NVEC];
        constExpr->GetValues(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT16: {
        uint16_t uv[ISPC_MAX_NVEC];
        constExpr->GetValues(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_INT32: {
        int32_t iv[ISPC_MAX_NVEC];
        constExpr->GetValues(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT32: {
        uint32_t uv[ISPC_MAX_NVEC];
        constExpr->GetValues(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_INT64: {
        int64_t iv[ISPC_MAX_NVEC];
        constExpr->GetValues(iv, forceVarying);
        return new ConstExpr(toType, iv, pos);
    }
    case AtomicType::TYPE_UINT64: {
        uint64_t uv[ISPC_MAX_NVEC];
        constExpr->GetValues(uv, forceVarying);
        return new ConstExpr(toType, uv, pos);
    }
    case AtomicType::TYPE_FLOAT16: {
        std::vector<llvm::APFloat> fh;
        constExpr->GetValues(fh, LLVMTypes::Float16Type, forceVarying);
        return new ConstExpr(toType, fh, pos);
    }
    case AtomicType::TYPE_FLOAT: {
        std::vector<llvm::APFloat> fv;
        constExpr->GetValues(fv, LLVMTypes::FloatType, forceVarying);
        return new ConstExpr(toType, fv, pos);
    }
    case AtomicType::TYPE_DOUBLE: {
        std::vector<llvm::APFloat> dv;
        constExpr->GetValues(dv, LLVMTypes::DoubleType, forceVarying);
        return new ConstExpr(toType, dv, pos);
    }
    default:
        FATAL("unimplemented");
    }
    return this;
}

int TypeCastExpr::EstimateCost() const {
    if (llvm::dyn_cast<ConstExpr>(expr) != nullptr)
        return 0;

    // FIXME: return COST_TYPECAST_COMPLEX when appropriate
    return COST_TYPECAST_SIMPLE;
}

TypeCastExpr *TypeCastExpr::Instantiate(TemplateInstantiation &templInst) const {
    const Type *instType = type ? type->ResolveDependenceForTopType(templInst) : nullptr;
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return new TypeCastExpr(instType, instExpr, pos);
}

void TypeCastExpr::Print(Indent &indent) const {
    indent.Print("TypeCastExpr", pos);
    printf("[%s]\n", GetType()->GetString().c_str());
    indent.pushSingle();
    expr->Print(indent);
    indent.Done();
}

Symbol *TypeCastExpr::GetBaseSymbol() const { return expr ? expr->GetBaseSymbol() : nullptr; }

static llvm::Constant *lConvertPointerConstant(llvm::Constant *c, const Type *constType) {
    if (c == nullptr || constType->IsUniformType())
        return c;

    // Handle conversion to int and then to vector of int or array of int
    // (for varying and soa types, respectively)
    llvm::Constant *intPtr = llvm::ConstantExpr::getPtrToInt(c, LLVMTypes::PointerIntType);
    Assert(constType->IsVaryingType() || constType->IsSOAType());
    int count = constType->IsVaryingType() ? g->target->getVectorWidth() : constType->GetSOAWidth();

    std::vector<llvm::Constant *> smear;
    for (int i = 0; i < count; ++i)
        smear.push_back(intPtr);

    if (constType->IsVaryingType())
        return llvm::ConstantVector::get(smear);
    else {
        llvm::ArrayType *at = llvm::ArrayType::get(LLVMTypes::PointerIntType, count);
        return llvm::ConstantArray::get(at, smear);
    }
}

std::pair<llvm::Constant *, bool> TypeCastExpr::GetConstant(const Type *constType) const {
    // We don't need to worry about most the basic cases where the type
    // cast can resolve to a constant here, since the
    // TypeCastExpr::Optimize() method generally ends up doing the type
    // conversion and returning a ConstExpr, which in turn will have its
    // GetConstant() method called.  However, because ConstExpr currently
    // can't represent pointer values, we have to handle a few cases
    // related to pointers here:
    //
    // 1. Null pointer (nullptr, 0) valued initializers
    // 2. Converting function types to pointer-to-function types
    // 3. And converting these from uniform to the varying/soa equivalents.
    //

    if ((CastType<PointerType>(constType) == nullptr) && (llvm::dyn_cast<SizeOfExpr>(expr) == nullptr))
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    llvm::Value *ptr = nullptr;
    if (GetBaseSymbol())
        ptr = GetBaseSymbol()->storageInfo ? GetBaseSymbol()->storageInfo->getPointer() : nullptr;

    if (ptr && llvm::dyn_cast<llvm::GlobalVariable>(ptr)) {
        if (CastType<ArrayType>(expr->GetType())) {
            if (llvm::Constant *c = llvm::dyn_cast<llvm::Constant>(ptr)) {
                llvm::Value *offsets[2] = {LLVMInt32(0), LLVMInt32(0)};
                llvm::ArrayRef<llvm::Value *> arrayRef(&offsets[0], &offsets[2]);
                llvm::Value *resultPtr = llvm::ConstantExpr::getGetElementPtr(
                    llvm::dyn_cast<llvm::GlobalVariable>(ptr)->getValueType(), c, arrayRef);
                if (resultPtr->getType() == constType->LLVMType(g->ctx)) {
                    llvm::Constant *ret = llvm::dyn_cast<llvm::Constant>(resultPtr);
                    return std::pair<llvm::Constant *, bool>(ret, false);
                }
            }
        }
    }

    std::pair<llvm::Constant *, bool> cPair = expr->GetConstant(constType->GetAsUniformType());
    llvm::Constant *c = cPair.first;
    return std::pair<llvm::Constant *, bool>(lConvertPointerConstant(c, constType), cPair.second);
}

///////////////////////////////////////////////////////////////////////////
// ReferenceExpr

ReferenceExpr::ReferenceExpr(Expr *e, SourcePos p) : Expr(p, ReferenceExprID) { expr = e; }

llvm::Value *ReferenceExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    if (expr == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    llvm::Value *value = expr->GetLValue(ctx);
    if (value != nullptr)
        return value;

    // value is nullptr if the expression is a temporary; in this case, we'll
    // allocate storage for it so that we can return the pointer to that...
    const Type *type;
    if ((type = expr->GetType()) == nullptr || type->LLVMType(g->ctx) == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    value = expr->GetValue(ctx);
    if (value == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    AddressInfo *ptrInfo = ctx->AllocaInst(type);
    ctx->StoreInst(value, ptrInfo, type);
    return ptrInfo->getPointer();
}

Symbol *ReferenceExpr::GetBaseSymbol() const { return expr ? expr->GetBaseSymbol() : nullptr; }

const Type *ReferenceExpr::GetType() const {
    if (!expr)
        return nullptr;

    const Type *type = expr->GetType();
    if (!type)
        return nullptr;

    if (type->IsDependentType()) {
        return AtomicType::Dependent;
    }

    return new ReferenceType(type);
}

const Type *ReferenceExpr::GetLValueType() const {
    if (!expr)
        return nullptr;

    const Type *type = expr->GetType();
    if (!type)
        return nullptr;

    return PointerType::GetUniform(type);
}

Expr *ReferenceExpr::Optimize() {
    if (expr == nullptr)
        return nullptr;
    return this;
}

Expr *ReferenceExpr::TypeCheck() {
    if (expr == nullptr)
        return nullptr;
    return this;
}

int ReferenceExpr::EstimateCost() const { return 0; }

ReferenceExpr *ReferenceExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return new ReferenceExpr(instExpr, pos);
}

void ReferenceExpr::Print(Indent &indent) const {
    if (expr == nullptr || GetType() == nullptr) {
        indent.Print("ReferenceExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("ReferenceExpr", pos);

    printf("[%s]\n", GetType()->GetString().c_str());
    indent.pushSingle();
    expr->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// DerefExpr

DerefExpr::DerefExpr(Expr *e, SourcePos p, unsigned scid) : Expr(p, scid) { expr = e; }

llvm::Value *DerefExpr::GetValue(FunctionEmitContext *ctx) const {
    if (expr == nullptr)
        return nullptr;
    llvm::Value *ptr = expr->GetValue(ctx);
    if (ptr == nullptr)
        return nullptr;
    const Type *type = expr->GetType();
    if (type == nullptr)
        return nullptr;

    if (lVaryingStructHasUniformMember(type, pos))
        return nullptr;

    // If dealing with 'varying * varying' add required offsets.
    ptr = lAddVaryingOffsetsIfNeeded(ctx, ptr, type);

    Symbol *baseSym = expr->GetBaseSymbol();
    llvm::Value *mask = baseSym ? lMaskForSymbol(baseSym, ctx) : ctx->GetFullMask();

    ctx->SetDebugPos(pos);
    return ctx->LoadInst(ptr, mask, type);
}

llvm::Value *DerefExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (expr == nullptr)
        return nullptr;
    return expr->GetValue(ctx);
}

const Type *DerefExpr::GetLValueType() const {
    if (expr == nullptr)
        return nullptr;
    return expr->GetType();
}

Symbol *DerefExpr::GetBaseSymbol() const { return expr ? expr->GetBaseSymbol() : nullptr; }

Expr *DerefExpr::Optimize() {
    if (expr == nullptr)
        return nullptr;
    return this;
}

///////////////////////////////////////////////////////////////////////////
// PtrDerefExpr

PtrDerefExpr::PtrDerefExpr(Expr *e, SourcePos p) : DerefExpr(e, p, PtrDerefExprID) {}

const Type *PtrDerefExpr::GetType() const {
    const Type *type;
    if (expr == nullptr || (type = expr->GetType()) == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }
    AssertPos(pos, CastType<PointerType>(type) != nullptr);

    if (type->IsDependentType()) {
        return AtomicType::Dependent;
    }

    if (type->IsUniformType())
        return type->GetBaseType();
    else
        return type->GetBaseType()->GetAsVaryingType();
}

Expr *PtrDerefExpr::TypeCheck() {
    const Type *type;
    if (expr == nullptr || (type = expr->GetType()) == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (type->IsDependentType()) {
        return this;
    }

    if (const PointerType *pt = CastType<PointerType>(type)) {
        if (pt->GetBaseType()->IsVoidType()) {
            Error(pos, "Illegal to dereference void pointer type \"%s\".", type->GetString().c_str());
            return nullptr;
        }
    } else {
        Error(pos, "Illegal to dereference non-pointer type \"%s\".", type->GetString().c_str());
        return nullptr;
    }

    return this;
}

int PtrDerefExpr::EstimateCost() const {
    const Type *type;
    if (expr == nullptr || (type = expr->GetType()) == nullptr) {
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

PtrDerefExpr *PtrDerefExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return new PtrDerefExpr(instExpr, pos);
}

void PtrDerefExpr::Print(Indent &indent) const {
    if (expr == nullptr || GetType() == nullptr) {
        indent.Print("PtrDerefExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("PtrDerefExpr", pos);

    printf("[%s]\n", GetType()->GetString().c_str());
    indent.pushSingle();
    expr->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// RefDerefExpr

RefDerefExpr::RefDerefExpr(Expr *e, SourcePos p) : DerefExpr(e, p, RefDerefExprID) {}

const Type *RefDerefExpr::GetType() const {
    const Type *type;
    if (expr == nullptr || (type = expr->GetType()) == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (type->IsDependentType()) {
        return AtomicType::Dependent;
    }

    AssertPos(pos, CastType<ReferenceType>(type) != nullptr);
    return type->GetReferenceTarget();
}

Expr *RefDerefExpr::TypeCheck() {
    const Type *type;
    if (expr == nullptr || (type = expr->GetType()) == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (type->IsDependentType()) {
        return this;
    }

    // We only create RefDerefExprs internally for references in
    // expressions, so we should never create one with a non-reference
    // expression...
    AssertPos(pos, CastType<ReferenceType>(type) != nullptr);

    return this;
}

int RefDerefExpr::EstimateCost() const {
    if (expr == nullptr)
        return 0;

    return COST_DEREF;
}

RefDerefExpr *RefDerefExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return new RefDerefExpr(instExpr, pos);
}

void RefDerefExpr::Print(Indent &indent) const {
    if (expr == nullptr || GetType() == nullptr) {
        indent.Print("RefDerefExpr: <NULL EXPR>\n");
        return;
    }

    indent.Print("RefDerefExpr", pos);

    printf("[%s]\n", GetType()->GetString().c_str());
    indent.pushSingle();
    expr->Print(indent);

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// AddressOfExpr

AddressOfExpr::AddressOfExpr(Expr *e, SourcePos p) : Expr(p, AddressOfExprID), expr(e) {}

llvm::Value *AddressOfExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    if (expr == nullptr)
        return nullptr;

    const Type *exprType = expr->GetType();
    if (CastType<ReferenceType>(exprType) != nullptr || CastType<FunctionType>(exprType) != nullptr)
        return expr->GetValue(ctx);
    else
        return expr->GetLValue(ctx);
}

const Type *AddressOfExpr::GetType() const {
    if (expr == nullptr)
        return nullptr;

    const Type *exprType = expr->GetType();
    if (exprType && exprType->IsDependentType()) {
        return AtomicType::Dependent;
    }

    if (CastType<ReferenceType>(exprType) != nullptr)
        return PointerType::GetUniform(exprType->GetReferenceTarget());

    const Type *t = expr->GetLValueType();
    if (t != nullptr)
        return t;
    else {
        t = expr->GetType();
        if (t == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }
        return PointerType::GetUniform(t);
    }
}

const Type *AddressOfExpr::GetLValueType() const {
    if (!expr)
        return nullptr;

    const Type *type = expr->GetType();
    if (!type)
        return nullptr;

    return PointerType::GetUniform(type);
}

Symbol *AddressOfExpr::GetBaseSymbol() const { return expr ? expr->GetBaseSymbol() : nullptr; }

void AddressOfExpr::Print(Indent &indent) const {
    if (expr == nullptr || GetType() == nullptr) {
        indent.Print("AddressOfExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("AddressOfExpr", pos);

    printf("[%s]\n", GetType()->GetString().c_str());
    indent.pushSingle();
    expr->Print(indent);

    indent.Done();
}

Expr *AddressOfExpr::TypeCheck() {
    const Type *exprType;
    if (expr == nullptr || (exprType = expr->GetType()) == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (exprType->IsDependentType()) {
        return this;
    }

    if (CastType<ReferenceType>(exprType) != nullptr || CastType<FunctionType>(exprType) != nullptr) {
        return this;
    }

    if (expr->GetLValueType() != nullptr)
        return this;

    Error(expr->pos, "Illegal to take address of non-lvalue or function.");
    return nullptr;
}

Expr *AddressOfExpr::Optimize() { return this; }

int AddressOfExpr::EstimateCost() const { return 0; }

AddressOfExpr *AddressOfExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return new AddressOfExpr(instExpr, pos);
}

std::pair<llvm::Constant *, bool> AddressOfExpr::GetConstant(const Type *type) const {
    if (expr == nullptr || expr->GetType() == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return std::pair<llvm::Constant *, bool>(nullptr, false);
    }

    const PointerType *pt = CastType<PointerType>(type);
    if (pt == nullptr)
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    bool isNotValidForMultiTargetGlobal = false;
    const FunctionType *ft = CastType<FunctionType>(pt->GetBaseType());
    if (ft != nullptr) {
        std::pair<llvm::Constant *, bool> cPair = expr->GetConstant(ft);
        llvm::Constant *c = cPair.first;
        return std::pair<llvm::Constant *, bool>(lConvertPointerConstant(c, type), cPair.second);
    }
    llvm::Value *ptr = nullptr;
    if (GetBaseSymbol())
        ptr = GetBaseSymbol()->storageInfo ? GetBaseSymbol()->storageInfo->getPointer() : nullptr;
    if (ptr && llvm::dyn_cast<llvm::GlobalVariable>(ptr)) {
        const Type *eTYPE = GetType();
        if (type->LLVMType(g->ctx) == eTYPE->LLVMType(g->ctx)) {
            if (llvm::dyn_cast<SymbolExpr>(expr) != nullptr) {
                return std::pair<llvm::Constant *, bool>(llvm::cast<llvm::Constant>(ptr),
                                                         isNotValidForMultiTargetGlobal);

            } else if (IndexExpr *IExpr = llvm::dyn_cast<IndexExpr>(expr)) {
                std::vector<llvm::Value *> gepIndex;
                Expr *mBaseExpr = nullptr;
                while (IExpr) {
                    std::pair<llvm::Constant *, bool> cIndexPair = IExpr->index->GetConstant(IExpr->index->GetType());
                    llvm::Constant *cIndex = cIndexPair.first;
                    if (cIndex == nullptr)
                        return std::pair<llvm::Constant *, bool>(nullptr, false);
                    gepIndex.insert(gepIndex.begin(), cIndex);
                    mBaseExpr = IExpr->baseExpr;
                    IExpr = llvm::dyn_cast<IndexExpr>(mBaseExpr);
                    isNotValidForMultiTargetGlobal = isNotValidForMultiTargetGlobal || cIndexPair.second;
                }
                // The base expression needs to be a global symbol so that the
                // address is a constant.
                if (llvm::dyn_cast<SymbolExpr>(mBaseExpr) == nullptr)
                    return std::pair<llvm::Constant *, bool>(nullptr, false);
                gepIndex.insert(gepIndex.begin(), LLVMInt64(0));
                llvm::Constant *c = llvm::cast<llvm::Constant>(ptr);
                llvm::Constant *c1 = llvm::ConstantExpr::getGetElementPtr(
                    llvm::dyn_cast<llvm::GlobalVariable>(ptr)->getValueType(), c, gepIndex);
                return std::pair<llvm::Constant *, bool>(c1, isNotValidForMultiTargetGlobal);
            }
        }
    }
    return std::pair<llvm::Constant *, bool>(nullptr, false);
}

///////////////////////////////////////////////////////////////////////////
// SizeOfExpr

SizeOfExpr::SizeOfExpr(Expr *e, SourcePos p) : Expr(p, SizeOfExprID), expr(e), type(nullptr) {}

SizeOfExpr::SizeOfExpr(const Type *t, SourcePos p) : Expr(p, SizeOfExprID), expr(nullptr), type(t) {
    type = type->ResolveUnboundVariability(Variability::Varying);
}

llvm::Value *SizeOfExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    const Type *t = expr ? expr->GetType() : type;
    if (t == nullptr)
        return nullptr;

    llvm::Type *llvmType = t->LLVMType(g->ctx);
    if (llvmType == nullptr)
        return nullptr;

    return g->target->SizeOf(llvmType, ctx->GetCurrentBasicBlock());
}

const Type *SizeOfExpr::GetType() const {
    return (g->target->is32Bit() || g->opt.force32BitAddressing) ? AtomicType::UniformUInt32
                                                                 : AtomicType::UniformUInt64;
}

void SizeOfExpr::Print(Indent &indent) const {
    indent.Print("SizeOfExpr", pos);

    if (expr != nullptr) {
        printf("\n");
        indent.pushSingle();
        expr->Print(indent);
    } else if (type != nullptr) {
        printf("type arg: [%s]\n", type->GetString().c_str());
    } else {
        printf("<NULL>\n");
    }

    indent.Done();
}

Expr *SizeOfExpr::TypeCheck() {
    if (type && type->IsDependentType()) {
        return this;
    }

    // Can't compute the size of a struct without a definition
    if (type != nullptr && CastType<UndefinedStructType>(type) != nullptr) {
        Error(pos,
              "Can't compute the size of declared but not defined "
              "struct type \"%s\".",
              type->GetString().c_str());
        return nullptr;
    }

    return this;
}

Expr *SizeOfExpr::Optimize() { return this; }

int SizeOfExpr::EstimateCost() const { return 0; }

SizeOfExpr *SizeOfExpr::Instantiate(TemplateInstantiation &templInst) const {
    if (expr != nullptr) {
        return new SizeOfExpr(expr->Instantiate(templInst), pos);
    }
    Assert(type != nullptr);
    return new SizeOfExpr(type->ResolveDependence(templInst), pos);
}

std::pair<llvm::Constant *, bool> SizeOfExpr::GetConstant(const Type *rtype) const {
    const Type *t = expr ? expr->GetType() : type;
    if (t == nullptr)
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    bool isNotValidForMultiTargetGlobal = false;
    if (t->IsVaryingType())
        isNotValidForMultiTargetGlobal = true;

    llvm::Type *llvmType = t->LLVMType(g->ctx);
    if (llvmType == nullptr)
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    uint64_t byteSize = g->target->getDataLayout()->getTypeStoreSize(llvmType);
    return std::pair<llvm::Constant *, bool>(llvm::ConstantInt::get(rtype->LLVMType(g->ctx), byteSize),
                                             isNotValidForMultiTargetGlobal);
}

///////////////////////////////////////////////////////////////////////////
// AllocaExpr

AllocaExpr::AllocaExpr(Expr *e, SourcePos p) : Expr(p, AllocaExprID), expr(e) {}

llvm::Value *AllocaExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    if (expr == nullptr)
        return nullptr;
    llvm::Value *llvmValue = expr->GetValue(ctx);
    if (llvmValue == nullptr)
        return nullptr;
    llvm::Value *resultPtr = ctx->AllocaInst(LLVMTypes::Int8Type, llvmValue, "allocaExpr", 16,
                                             false)
                                 ->getPointer(); // 16 byte stack alignment.
    return resultPtr;
}

const Type *AllocaExpr::GetType() const { return PointerType::Void; }

void AllocaExpr::Print(Indent &indent) const {
    if (expr == nullptr) {
        indent.Print("AllocaExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.PrintLn("AllocaExpr", pos);

    indent.pushSingle();
    expr->Print(indent);

    indent.Done();
}

Expr *AllocaExpr::TypeCheck() {
    if (expr == nullptr) {
        return nullptr;
    }

    const Type *argType = expr ? expr->GetType() : nullptr;
    if (argType && argType->IsDependentType()) {
        return this;
    }

    const Type *sizeType = m->symbolTable->LookupType("size_t");
    Assert(sizeType != nullptr);
    if (!Type::Equal(sizeType->GetAsUniformType(), expr->GetType())) {
        expr = TypeConvertExpr(expr, sizeType->GetAsUniformType(), "Alloca_arg");
    }
    if (expr == nullptr) {
        Assert(argType);
        Error(pos, "\"alloca()\" cannot have an argument of type \"%s\".", argType->GetString().c_str());
        return nullptr;
    }

    return this;
}

Expr *AllocaExpr::Optimize() { return this; }

int AllocaExpr::EstimateCost() const { return 0; }

AllocaExpr *AllocaExpr::Instantiate(TemplateInstantiation &templInst) const {
    Expr *instExpr = expr ? expr->Instantiate(templInst) : nullptr;
    return new AllocaExpr(instExpr, pos);
}

///////////////////////////////////////////////////////////////////////////
// SymbolExpr

SymbolExpr::SymbolExpr(Symbol *s, SourcePos p) : Expr(p, SymbolExprID) { symbol = s; }

llvm::Value *SymbolExpr::GetValue(FunctionEmitContext *ctx) const {
    // storageInfo may be nullptr due to an earlier compilation error
    if (!symbol || !symbol->storageInfo)
        return nullptr;
    ctx->SetDebugPos(pos);

    std::string loadName = symbol->name + std::string("_load");
#ifdef ISPC_XE_ENABLED
    // TODO: this is a temporary workaround and will be changed as part
    // of SPIR-V emitting solution
    if (ctx->emitXeHardwareMask() && symbol->name == "__mask") {
        return ctx->XeSimdCFPredicate(LLVMMaskAllOn);
    }
#endif
    return ctx->LoadInst(symbol->storageInfo, symbol->type, loadName.c_str());
}

llvm::Value *SymbolExpr::GetLValue(FunctionEmitContext *ctx) const {
    if (symbol == nullptr)
        return nullptr;
    ctx->SetDebugPos(pos);
    return symbol->storageInfo->getPointer();
}

const Type *SymbolExpr::GetLValueType() const {
    if (symbol == nullptr)
        return nullptr;

    if (CastType<ReferenceType>(symbol->type) != nullptr)
        return PointerType::GetUniform(symbol->type->GetReferenceTarget());
    else
        return PointerType::GetUniform(symbol->type);
}

Symbol *SymbolExpr::GetBaseSymbol() const { return symbol; }

const Type *SymbolExpr::GetType() const { return symbol ? symbol->type : nullptr; }

Expr *SymbolExpr::TypeCheck() { return this; }

Expr *SymbolExpr::Optimize() {
    if (symbol == nullptr)
        return nullptr;
    else if (symbol->constValue != nullptr) {
        AssertPos(pos, GetType()->IsConstType());
        return new ConstExpr(symbol->constValue, pos);
    } else
        return this;
}

int SymbolExpr::EstimateCost() const {
    // Be optimistic and assume it's in a register or can be used as a
    // memory operand..
    return 0;
}

SymbolExpr *SymbolExpr::Instantiate(TemplateInstantiation &templInst) const {
    Symbol *resolvedSymbol = templInst.InstantiateSymbol(symbol);
    return new SymbolExpr(resolvedSymbol, pos);
}

void SymbolExpr::Print(Indent &indent) const {
    if (symbol == nullptr || GetType() == nullptr) {
        indent.Print("SymbolExpr: <NULL EXPR>\n");
        indent.Done();
        return;
    }

    indent.Print("SymbolExpr", pos);

    printf("[%s] symbol name: %s\n", GetType()->GetString().c_str(), symbol->name.c_str());

    indent.Done();
}

///////////////////////////////////////////////////////////////////////////
// FunctionSymbolExpr

FunctionSymbolExpr::FunctionSymbolExpr(const char *n, const std::vector<Symbol *> &candidates, SourcePos p)
    : Expr(p, FunctionSymbolExprID), name(n), candidateFunctions(candidates), triedToResolve(false),
      unresolvedButDependent(false) {
    matchingFunc = (candidates.size() == 1) ? candidates[0] : nullptr;
}

FunctionSymbolExpr::FunctionSymbolExpr(const char *n, const std::vector<TemplateSymbol *> &candidates,
                                       const TemplateArgs &templArgs, SourcePos p)
    : Expr(p, FunctionSymbolExprID), name(n), candidateTemplateFunctions(candidates), templateArgs(templArgs),
      matchingFunc(nullptr), triedToResolve(false), unresolvedButDependent(false) {
    // Do template argument "normalization", i.e apply "varying type default":
    //
    // template <typename T> void foo(T t);
    // foo<int>(1); // T is assumed to be "varying int" here.
    for (auto &arg : templateArgs) {
        arg.SetAsVaryingType();
    }
}

const Type *FunctionSymbolExpr::GetType() const {
    if (unresolvedButDependent) {
        return AtomicType::Dependent;
    }

    if (triedToResolve == false && matchingFunc == nullptr) {
        Error(pos, "Ambiguous use of overloaded function \"%s\".", name.c_str());
        return nullptr;
    }

    return matchingFunc ? matchingFunc->type : nullptr;
}

llvm::Value *FunctionSymbolExpr::GetValue(FunctionEmitContext *ctx) const {
    return matchingFunc ? matchingFunc->function : nullptr;
}

Symbol *FunctionSymbolExpr::GetBaseSymbol() const { return matchingFunc; }

Expr *FunctionSymbolExpr::TypeCheck() { return this; }

Expr *FunctionSymbolExpr::Optimize() { return this; }

int FunctionSymbolExpr::EstimateCost() const { return 0; }

FunctionSymbolExpr *FunctionSymbolExpr::Instantiate(TemplateInstantiation &templInst) const {
    // TODO: interfaces for regular function call and template function call should be unified.
    // Bonus: it is possbile that a call can possbily refer to both.
    if (candidateFunctions.size() != 0) {
        Assert(candidateTemplateFunctions.size() == 0);
        return new FunctionSymbolExpr(name.c_str(), candidateFunctions, pos);
    }
    TemplateArgs instTemplateArgs;
    for (auto &arg : templateArgs) {
        instTemplateArgs.push_back(
            arg.IsType() ? TemplateArg(arg.GetAsType()->ResolveDependenceForTopType(templInst), arg.GetPos())
                         : TemplateArg(arg.GetAsExpr()->Instantiate(templInst), arg.GetPos()));
    }
    return new FunctionSymbolExpr(name.c_str(), candidateTemplateFunctions, instTemplateArgs, pos);
}

void FunctionSymbolExpr::Print(Indent &indent) const {
    const Type *type = GetType();

    indent.Print("FunctionSymbolExpr", pos);

    if (type && type->IsDependentType()) {
        printf("[%s] %s\n", type->GetString().c_str(), name.c_str());
    } else if (!matchingFunc || !type) {
        indent.Print("FunctionSymbolExpr: <NULL EXPR>\n");
    } else {
        printf("[%s] function name: %s\n", type->GetString().c_str(), matchingFunc->name.c_str());
    }

    indent.Done();
}

std::pair<llvm::Constant *, bool> FunctionSymbolExpr::GetConstant(const Type *type) const {
    if (matchingFunc == nullptr || matchingFunc->function == nullptr)
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    const FunctionType *ft = CastType<FunctionType>(type);
    if (ft == nullptr)
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    if (Type::Equal(type, matchingFunc->type) == false) {
        Error(pos,
              "Type of function symbol \"%s\" doesn't match expected type "
              "\"%s\".",
              matchingFunc->type->GetString().c_str(), type->GetString().c_str());
        return std::pair<llvm::Constant *, bool>(nullptr, false);
    }

    return std::pair<llvm::Constant *, bool>(matchingFunc->function, false);
}

static std::string lGetOverloadCandidateMessage(const std::vector<Symbol *> &funcs,
                                                const std::vector<const Type *> &argTypes,
                                                const std::vector<bool> *argCouldBeNULL) {
    std::string message = "Passed types: (";
    for (unsigned int i = 0; i < argTypes.size(); ++i) {
        if (argTypes[i] != nullptr)
            message += argTypes[i]->GetString();
        else
            message += "(unknown type)";
        message += (i < argTypes.size() - 1) ? ", " : ")\n";
    }

    for (unsigned int i = 0; i < funcs.size(); ++i) {
        const FunctionType *ft = CastType<FunctionType>(funcs[i]->type);
        Assert(ft != nullptr);
        message += "Candidate: ";
        message += ft->GetString();
        if (i < funcs.size() - 1)
            message += "\n";
    }
    return message;
}

/** Helper function used for function overload resolution: returns true if
    converting the argument to the call type only requires a type
    conversion that won't lose information.  Otherwise return false.
  */
static bool lIsMatchWithTypeWidening(const Type *callType, const Type *funcArgType) {
    const AtomicType *callAt = CastType<AtomicType>(callType);
    const AtomicType *funcAt = CastType<AtomicType>(funcArgType);
    if (callAt == nullptr || funcAt == nullptr)
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
        return (funcAt->basicType != AtomicType::TYPE_BOOL && funcAt->basicType != AtomicType::TYPE_INT8 &&
                funcAt->basicType != AtomicType::TYPE_UINT8);
    case AtomicType::TYPE_INT32:
    case AtomicType::TYPE_UINT32:
        return (funcAt->basicType == AtomicType::TYPE_INT32 || funcAt->basicType == AtomicType::TYPE_UINT32 ||
                funcAt->basicType == AtomicType::TYPE_INT64 || funcAt->basicType == AtomicType::TYPE_UINT64);
    case AtomicType::TYPE_FLOAT16:
        return (funcAt->basicType == AtomicType::TYPE_FLOAT || funcAt->basicType == AtomicType::TYPE_DOUBLE);
    case AtomicType::TYPE_FLOAT:
        return (funcAt->basicType == AtomicType::TYPE_DOUBLE);
    case AtomicType::TYPE_INT64:
    case AtomicType::TYPE_UINT64:
        return (funcAt->basicType == AtomicType::TYPE_INT64 || funcAt->basicType == AtomicType::TYPE_UINT64);
    case AtomicType::TYPE_DOUBLE:
        return false;
    default:
        FATAL("Unhandled atomic type");
        return false;
    }
}

/* Returns the set of function overloads that are potential matches, given
   argCount values being passed as arguments to the function call.
 */
std::vector<Symbol *> FunctionSymbolExpr::getCandidateFunctions(int argCount) const {
    std::vector<Symbol *> ret;
    for (Symbol *sym : candidateFunctions) {
        AssertPos(pos, sym != nullptr);
        const FunctionType *ft = CastType<FunctionType>(sym->type);
        AssertPos(pos, ft != nullptr);

        // There's no way to match if the caller is passing more arguments
        // than this function instance takes.
        if (argCount > ft->GetNumParameters()) {
            continue;
        }

        // Not enough arguments, and no default argument value to save us
        if (argCount < ft->GetNumParameters() && ft->GetParameterDefault(argCount) == nullptr) {
            continue;
        }

        // Success
        ret.push_back(sym);
    }
    return ret;
}

// Return deduced parameter (a pair of parameter name and corresponding type). In case of failure,
// return a pair of empty strings and a nullptr type.
static std::pair<std::string, const Type *> lDeduceParam(const Type *paramType, const Type *argType) {
    if (paramType == nullptr || argType == nullptr) {
        return std::pair<std::string, const Type *>("", nullptr);
    }

    bool constParam = paramType->IsConstType() && !paramType->IsReferenceType();
    bool constArg = argType->IsConstType() && !argType->IsReferenceType();

    // Const
    if (constParam && constArg) {
        return lDeduceParam(paramType->GetAsNonConstType(), argType->GetAsNonConstType());
    }

    // Reference
    if (paramType->IsReferenceType() && argType->IsReferenceType()) {
        return lDeduceParam(paramType->GetReferenceTarget(), argType->GetReferenceTarget());
    }

    // Pointers
    if (paramType->IsPointerType() && argType->IsPointerType()) {
        // No additional restrictions for pointer const and variability matching, if they don't match,
        // the candidate will be rejected by conversion rules.
        return lDeduceParam(paramType->GetBaseType(), argType->GetBaseType());
    }

    const TemplateTypeParmType *templTypeParam = CastType<TemplateTypeParmType>(paramType);
    if (templTypeParam) {
        // template <typename T> void foo(T t);
        // template <typename T> void foo(uniform T t);
        // template <typename T> void foo(varying T t);
        //
        // In case of uniform/varying arguments the deduction rules will be the following:
        //
        // argument |     T t     | uniform T t | varying T t
        // --------------------------------------------------
        // uniform  | T = uniform | T = varying | T = uniform
        // varying  | T = varying | error       | T = uniform

        const Type *deducedType = argType;

        if (templTypeParam->IsUniformType() && argType->IsVaryingType()) {
            // TODO: report a proper log message, which should be used in case of not matching overload is found.
            return std::pair<std::string, const Type *>("", nullptr);
        } else if (templTypeParam->IsUniformType() && argType->IsUniformType()) {
            deducedType = argType->GetAsVaryingType();
        } else if (templTypeParam->IsVaryingType() && argType->IsVaryingType()) {
            deducedType = argType->GetAsUniformType();
        }

        Assert(deducedType->GetVariability() != Variability::Unbound);

        return std::pair<std::string, const Type *>(templTypeParam->GetName(), deducedType);
    }

    return std::pair<std::string, const Type *>("", nullptr);
}

std::vector<Symbol *>
FunctionSymbolExpr::getCandidateTemplateFunctions(const std::vector<const Type *> &argTypes) const {
    // Two different cases are possible here:
    // 1. The template function was specified as simple_template_id, i.e. with explicit template arguments:
    //      foo<int>(arg1, arg2);
    //    In this case no type deduction is needed - the types were explicitly specified.
    //    The arguments still need to be verified for viability - i.e. that Implicit Convertion Sequence (ICS)
    //    exists (and is better that others).
    // 2. The template function was specified by name, without tempalte paramters, i.e.
    //      foo(arg1, arg2);
    //    In this case template type parameters deduction need to happen.
    //    And then the same step for ICS to be done for the candidate.
    //

    std::vector<Symbol *> ret;
    for (TemplateSymbol *templSym : candidateTemplateFunctions) {
        AssertPos(pos, templSym != nullptr);
        const FunctionType *ft = CastType<FunctionType>(templSym->type);
        AssertPos(pos, ft != nullptr);

        // There's no way to match if the caller is passing more arguments
        // than this function instance takes.
        if (argTypes.size() > ft->GetNumParameters()) {
            continue;
        }

        // Not enough arguments, and no default argument value to save us
        if (argTypes.size() < ft->GetNumParameters() && ft->GetParameterDefault(argTypes.size()) == nullptr) {
            continue;
        }

        // Terminology used below:
        // template <typename T> void foo(T a);
        //           ^^^^^^^^^^           ^^^
        //           template parameters  function parameters
        //           (templateParms)      (paramTypes)
        //
        // foo<int>(1);
        //     ^^^  ^
        //     |||  function arguments
        //     |||  (argTypes)
        //     |||
        //     template arguments (explicitly specified)
        //     (templateArgs)
        //

        if (!templSym->functionTemplate) {
            // This template function was only declared, but not defined.
            // TODO: make it work for declared functions too.
            continue;
        }

        // Template parameters
        const TemplateParms *templateParms = templSym->functionTemplate->GetTemplateParms();

        // Function parameters
        const FunctionType *ftype = templSym->functionTemplate->GetFunctionType();
        std::vector<const Type *> paramTypes;
        for (int i = 0; i < ftype->GetNumParameters(); ++i) {
            paramTypes.push_back(ft->GetParameterType(i));
        }

        // This looks like a candidate, so now we need get to instantiation and add it to candidate list.
        if (templateArgs.size() == templateParms->GetCount()) {
            // Easy, we have all template arguments specified explicitly, no deduction is needed.
            // First, check types of non-type parameters (non-type parameters can't be used in partially specified
            // template instantiations)
            bool argsMatchingPassed = true;
            for (int i = 0; i < templateParms->GetCount(); ++i) {
                if ((*templateParms)[i]->IsNonTypeParam()) {
                    const Type *argType = templateArgs[i].GetAsType();
                    const Type *paramType = (*templateParms)[i]->GetNonTypeParam()->type;
                    if (!CanConvertTypes(argType, paramType)) {
                        argsMatchingPassed = false;
                        break;
                    }
                }
            }
            if (!argsMatchingPassed) {
                continue;
            }
            Symbol *funcSym = templSym->functionTemplate->LookupInstantiation(templateArgs);
            if (funcSym == nullptr) {
                funcSym = templSym->functionTemplate->AddInstantiation(
                    templateArgs, TemplateInstantiationKind::Implicit, templSym->isInline, templSym->isNoInline);
            }
            AssertPos(pos, funcSym);
            // Success
            ret.push_back(funcSym);
            continue;
        } else if (templateArgs.size() > templateParms->GetCount()) {
            // Too many template arguments specified
            continue;
        }

        // Create substitution map for specified template parameters
        TemplateInstantiation inst(*templateParms, templateArgs, TemplateInstantiationKind::Implicit,
                                   templSym->isInline, templSym->isNoInline);

        std::vector<const Type *> substitutedParamTypes;
        // Instantiate function parameter types with explicitly specified template arguments
        for (const Type *paramType : paramTypes) {
            substitutedParamTypes.push_back(paramType->ResolveDependence(inst));
        }

        bool deductionFailed = false;

        // Deduce template parameters from function arguments. Trying to follow C++ template argument deduction
        // algorithm.
        for (int i = 0; i < substitutedParamTypes.size(); ++i) {
            const Type *paramType = substitutedParamTypes[i];
            if (paramType->IsDependentType()) {
                // Try to deduce

                // For the sake of template argument deduction, argument cannot have a reference type, as
                // "In C++, the type of an expression is always adjusted so that it will not have reference type (C++
                // [expr]p6)"
                const Type *argType = argTypes[i];
                if (argType->IsReferenceType()) {
                    argType = CastType<ReferenceType>(argType)->GetReferenceTarget();
                }

                // Note that in C++ `const T &` is a reference type, but not a constant type. Stripping cv-qualifiers
                // from this type has no effect, but stripping reference type yields `cosnt T`. ISPC type system treats
                // `const T &` as both constant and reference type, but stripping either of them keeps the other
                // property. We need to mimic C++ behavior here.

                if (!paramType->IsReferenceType()) {
                    paramType = paramType->GetAsNonConstType();

                    // If P is not a reference:
                    // - if A is an array type, apply array-to-pointer conversion.
                    const ArrayType *at = CastType<ArrayType>(argType);
                    if (at) {
                        const Type *targetType = at->GetElementType();
                        if (targetType == nullptr) {
                            deductionFailed = true;
                            break;
                        }
                        argType = PointerType::GetUniform(targetType, at->IsSOAType());
                    }
                    // TODO: - if A is a function type, apply function-to-pointer conversion.
                    // - if A is a cv-qualified type, remove the top level cv-qualifiers from A.
                    argType = argType->GetAsNonConstType();
                } else {
                    // If P is a reference type, remove the reference part of P.
                    paramType = CastType<ReferenceType>(paramType)->GetReferenceTarget();
                }

                // Deduce. The result is a pair of template parameter name and type.
                auto deduction = lDeduceParam(paramType, argType);
                if (deduction.second != nullptr) {
                    // check if this deduction is compatible with previously deduced arguments.
                    const Type *previousDeductionResult = inst.InstantiateType(deduction.first);

                    if (previousDeductionResult == nullptr) {
                        // This tempalte parameter was deducted for the first time. Add it to the map.
                        inst.AddArgument(deduction.first, TemplateArg(deduction.second, pos));
                    } else if (!Type::Equal(previousDeductionResult, deduction.second)) {
                        if (previousDeductionResult->IsUniformType() && deduction.second->IsVaryingType() &&
                            Type::Equal(previousDeductionResult->GetAsVaryingType(), deduction.second)) {
                            // override previous deduction with varying type
                            inst.AddArgument(deduction.first, TemplateArg(deduction.second, pos));
                        } else if (previousDeductionResult->IsVaryingType() && deduction.second->IsUniformType()) {
                            // That's fine, uniform will be broadcasted.
                        } else {
                            // Deduction failed due to conflicting deduction types.
                            deductionFailed = true;
                            break;
                        }
                    } else {
                        // Deducted type is the same as previously deduced one. Do nothing, we are good.
                    }
                } else {
                    // Deduction failed, skip to the next candidate.
                    deductionFailed = true;
                    break;
                }
            }
        }

        if (deductionFailed) {
            continue;
        }

        // Build a complete vector of deduced template arguments.
        TemplateArgs deducedArgs;
        for (int i = 0; i < templateParms->GetCount(); ++i) {
            if (i < templateArgs.size()) {
                deducedArgs.push_back(templateArgs[i]);
            } else {
                const Type *deducedArg = inst.InstantiateType((*templateParms)[i]->GetName());
                if (!deducedArg || deducedArg->IsDependentType()) {
                    // Undeduced template parameter. The deduction is incomplete and we need to skip to another
                    // candidate.
                    deductionFailed = true;
                    break;
                }
                deducedArgs.push_back(TemplateArg(deducedArg, pos));
            }
        }
        if (deductionFailed) {
            continue;
        }

        // All template arguments were either explicitly specified or deduced, now get the instantiation.
        Symbol *funcSym = templSym->functionTemplate->LookupInstantiation(deducedArgs);
        if (funcSym == nullptr) {
            funcSym = templSym->functionTemplate->AddInstantiation(deducedArgs, TemplateInstantiationKind::Implicit,
                                                                   templSym->isInline, templSym->isNoInline);
        }
        AssertPos(pos, funcSym);
        // Success
        ret.push_back(funcSym);
    }
    return ret;
}

static bool lArgIsPointerType(const Type *type) {
    if (CastType<PointerType>(type) != nullptr)
        return true;

    const ReferenceType *rt = CastType<ReferenceType>(type);
    if (rt == nullptr)
        return false;

    const Type *t = rt->GetReferenceTarget();
    return (CastType<PointerType>(t) != nullptr);
}

/** This function computes the value of a cost function that represents the
    cost of calling a function of the given type with arguments of the
    given types.  If it's not possible to call the function, regardless of
    any type conversions applied, a cost of -1 is returned.
 */
int FunctionSymbolExpr::computeOverloadCost(const FunctionType *ftype, const std::vector<const Type *> &argTypes,
                                            const std::vector<bool> *argCouldBeNULL,
                                            const std::vector<bool> *argIsConstant, int *cost) {
    int costSum = 0;

    // In computing the cost function, we only worry about the actual
    // argument types--using function default parameter values is free for
    // the purposes here...
    for (int i = 0; i < (int)argTypes.size(); ++i) {
        cost[i] = 0;
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
            // Step "1" from documentation
            cost[i] += 0;
        else if (argCouldBeNULL && (*argCouldBeNULL)[i] && lArgIsPointerType(fargType))
            // Passing nullptr to a pointer-typed parameter is also a no-cost operation
            // Step "1" from documentation
            cost[i] += 0;
        else {
            // If the argument is a compile-time constant, we'd like to
            // count the cost of various conversions as much lower than the
            // cost if it wasn't--so scale up the cost when this isn't the
            // case..
            if (argIsConstant == nullptr || (*argIsConstant)[i] == false)
                costScale *= 512;

            if (CastType<ReferenceType>(fargType)) {
                // Here we completely handle the case where fargType is reference.
                if (callType->IsConstType() && !fargType->IsConstType()) {
                    // It is forbidden to pass const object to non-const reference (cvf -> vfr)
                    return -1;
                }
                if (!callType->IsConstType() && fargType->IsConstType()) {
                    // It is possible to pass (vf -> cvfr)
                    // but it is worse than (vf -> vfr) or (cvf -> cvfr)
                    // Step "3" from documentation
                    cost[i] += 2 * costScale;
                }
                if (!Type::Equal(callType->GetReferenceTarget()->GetAsNonConstType(),
                                 fargType->GetReferenceTarget()->GetAsNonConstType())) {
                    // Types under references must be equal completely.
                    // vd -> vfr or vd -> cvfr are forbidden. (Although clang allows vd -> cvfr case.)
                    return -1;
                }
                // penalty for equal types under reference (vf -> vfr is worse than vf -> vf)
                // Step "2" from documentation
                cost[i] += 2 * costScale;
                continue;
            }
            const Type *callTypeNP = callType;
            if (CastType<ReferenceType>(callType)) {
                callTypeNP = callType->GetReferenceTarget();
                // we can treat vfr as vf for callType with some penalty
                // Step "5" from documentation
                cost[i] += 2 * costScale;
            }

            // Now we deal with references, so we can normalize to non-const types
            // because we're passing by value anyway, so const doesn't matter.
            const Type *callTypeNC = callTypeNP->GetAsNonConstType();
            const Type *fargTypeNC = fargType->GetAsNonConstType();

            // Now we forget about constants and references!
            if (Type::EqualIgnoringConst(callTypeNP, fargType)) {
                // The best case: vf -> vf.
                // Step "4" from documentation
                cost[i] += 1 * costScale;
                continue;
            }
            if (lIsMatchWithTypeWidening(callTypeNC, fargTypeNC)) {
                // A little bit worse case: vf -> vd.
                // Step "6" from documentation
                cost[i] += 8 * costScale;
                continue;
            }
            if (fargType->IsVaryingType() && callType->IsUniformType()) {
                // Here we deal with brodcasting uniform to varying.
                // callType - varying and fargType - uniform is forbidden.
                if (Type::Equal(callTypeNC->GetAsVaryingType(), fargTypeNC)) {
                    // uf -> vf is better than uf -> ui or uf -> ud
                    // Step "7" from documentation
                    cost[i] += 16 * costScale;
                    continue;
                }
                if (lIsMatchWithTypeWidening(callTypeNC->GetAsVaryingType(), fargTypeNC)) {
                    // uf -> vd is better than uf -> vi (128 < 128 + 64)
                    // but worse than uf -> ui (128 > 64)
                    // Step "9" from documentation
                    cost[i] += 128 * costScale;
                    continue;
                }
                // 128 + 64 is the max. uf -> vi is the worst case.
                // Step "10" from documentation
                cost[i] += 128 * costScale;
            }
            if (CanConvertTypes(callTypeNC, fargTypeNC))
                // two cases: the worst is 128 + 64: uf -> vi and
                // the only 64: (64 < 128) uf -> ui worse than uf -> vd
                // Step "8" from documentation
                cost[i] += 64 * costScale;
            else
                // Failure--no type conversion possible...
                return -1;
        }
    }

    for (int i = 0; i < (int)argTypes.size(); ++i) {
        costSum = costSum + cost[i];
    }
    return costSum;
}

bool FunctionSymbolExpr::ResolveOverloads(SourcePos argPos, const std::vector<const Type *> &argTypes,
                                          const std::vector<bool> *argCouldBeNULL,
                                          const std::vector<bool> *argIsConstant) {
    // Note: this name may be different from "name" only for intrinsics and this is
    // probably also wrong.
    // const char *funName = candidateFunctions.front()->name.c_str();
    if (triedToResolve == true) {
        return true;
    }
    triedToResolve = true;

    for (auto argType : argTypes) {
        if (argType->IsDependentType()) {
            unresolvedButDependent = true;
            break;
        }
    }
    if (unresolvedButDependent) {
        return true;
    }

    // First, find the subset of overload candidates that take the same
    // number of arguments as have parameters (including functions that
    // take more arguments but have defaults starting no later than after
    // our last parameter).
    std::vector<Symbol *> actualCandidates = getCandidateFunctions(argTypes.size());
    std::vector<Symbol *> templateCandidates = getCandidateTemplateFunctions(argTypes);
    actualCandidates.insert(actualCandidates.end(), templateCandidates.begin(), templateCandidates.end());

    int bestMatchCost = 1 << 30;
    std::vector<Symbol *> matches;
    std::vector<int> candidateCosts;
    std::vector<int *> candidateExpandCosts;

    if (actualCandidates.size() == 0)
        goto failure;

    // Compute the cost for calling each of the candidate functions
    for (int i = 0; i < (int)actualCandidates.size(); ++i) {
        const FunctionType *ft = CastType<FunctionType>(actualCandidates[i]->type);
        AssertPos(pos, ft != nullptr);
        int *cost = new int[argTypes.size()];
        candidateCosts.push_back(computeOverloadCost(ft, argTypes, argCouldBeNULL, argIsConstant, cost));
        candidateExpandCosts.push_back(cost);
    }

    // Find the best cost, and then the candidate or candidates that have
    // that cost.
    for (int i = 0; i < (int)candidateCosts.size(); ++i) {
        if (candidateCosts[i] != -1 && candidateCosts[i] < bestMatchCost)
            bestMatchCost = candidateCosts[i];
    }
    // None of the candidates matched
    if (bestMatchCost == (1 << 30))
        goto failure;
    for (int i = 0; i < (int)candidateCosts.size(); ++i) {
        if (candidateCosts[i] == bestMatchCost) {
            for (int j = 0; j < (int)candidateCosts.size(); ++j) {
                for (int k = 0; k < argTypes.size(); k++) {
                    if (candidateCosts[j] != -1 && candidateExpandCosts[j][k] < candidateExpandCosts[i][k]) {
                        std::vector<Symbol *> temp;
                        temp.push_back(actualCandidates[i]);
                        temp.push_back(actualCandidates[j]);
                        std::string candidateMessage = lGetOverloadCandidateMessage(temp, argTypes, argCouldBeNULL);
                        Warning(pos,
                                "call to \"%s\" is ambiguous. "
                                "This warning will be turned into error in the next ispc release.\n"
                                "Please add explicit cast to arguments to have unambiguous match."
                                "\n%s",
                                name.c_str(), candidateMessage.c_str());
                    }
                }
            }
            matches.push_back(actualCandidates[i]);
        }
    }
    for (int i = 0; i < (int)candidateExpandCosts.size(); ++i) {
        delete[] candidateExpandCosts[i];
    }

    if (matches.size() == 1) {
        // Only one match: success
        matchingFunc = matches[0];
        return true;
    } else if (matches.size() > 1) {
        // Multiple matches: ambiguous
        std::string candidateMessage = lGetOverloadCandidateMessage(matches, argTypes, argCouldBeNULL);
        Error(pos, "Multiple overloaded functions matched call to function \"%s\".\n%s", name.c_str(),
              candidateMessage.c_str());
        return false;
    } else {
        // No matches at all
    failure:
        std::string candidateMessage = lGetOverloadCandidateMessage(matches, argTypes, argCouldBeNULL);
        Error(pos, "Unable to find any matching overload for call to function \"%s\".\n%s", name.c_str(),
              candidateMessage.c_str());
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////
// SyncExpr

const Type *SyncExpr::GetType() const { return AtomicType::Void; }

llvm::Value *SyncExpr::GetValue(FunctionEmitContext *ctx) const {
    ctx->SetDebugPos(pos);
    ctx->SyncInst();
    return nullptr;
}

int SyncExpr::EstimateCost() const { return COST_SYNC; }

SyncExpr *SyncExpr::Instantiate(TemplateInstantiation &templInst) const { return new SyncExpr(pos); }

void SyncExpr::Print(Indent &indent) const {
    indent.PrintLn("SyncExpr", pos);
    indent.Done();
}

Expr *SyncExpr::TypeCheck() { return this; }

Expr *SyncExpr::Optimize() { return this; }

///////////////////////////////////////////////////////////////////////////
// NullPointerExpr

llvm::Value *NullPointerExpr::GetValue(FunctionEmitContext *ctx) const {
    return llvm::ConstantPointerNull::get(LLVMTypes::VoidPointerType);
}

const Type *NullPointerExpr::GetType() const { return PointerType::Void; }

Expr *NullPointerExpr::TypeCheck() { return this; }

Expr *NullPointerExpr::Optimize() { return this; }

std::pair<llvm::Constant *, bool> NullPointerExpr::GetConstant(const Type *type) const {
    const PointerType *pt = CastType<PointerType>(type);
    if (pt == nullptr)
        return std::pair<llvm::Constant *, bool>(nullptr, false);

    llvm::Type *llvmType = type->LLVMType(g->ctx);
    if (llvmType == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return std::pair<llvm::Constant *, bool>(nullptr, false);
    }

    return std::pair<llvm::Constant *, bool>(llvm::Constant::getNullValue(llvmType), false);
}

void NullPointerExpr::Print(Indent &indent) const {
    indent.PrintLn("NullPointerExpr", pos);
    indent.Done();
}

int NullPointerExpr::EstimateCost() const { return 0; }

NullPointerExpr *NullPointerExpr::Instantiate(TemplateInstantiation &templInst) const {
    return new NullPointerExpr(pos);
}

///////////////////////////////////////////////////////////////////////////
// NewExpr

NewExpr::NewExpr(int typeQual, const Type *t, Expr *init, Expr *count, SourcePos tqPos, SourcePos p)
    : Expr(p, NewExprID) {
    allocType = t;

    initExpr = init;
    countExpr = count;

    /* (The below cases actually should be impossible, since the parser
       doesn't allow more than a single type qualifier before a "new".) */
    if ((typeQual & ~(TYPEQUAL_UNIFORM | TYPEQUAL_VARYING)) != 0) {
        Error(tqPos, "Illegal type qualifiers in \"new\" expression (only "
                     "\"uniform\" and \"varying\" are allowed.");
        isVarying = false;
    } else if ((typeQual & TYPEQUAL_UNIFORM) != 0 && (typeQual & TYPEQUAL_VARYING) != 0) {
        Error(tqPos, "Illegal to provide both \"uniform\" and \"varying\" "
                     "qualifiers to \"new\" expression.");
        isVarying = false;
    } else
        // If no type qualifier is given before the 'new', treat it as a
        // varying new.
        isVarying = (typeQual == 0) || (typeQual & TYPEQUAL_VARYING);

    if (allocType != nullptr)
        allocType = allocType->ResolveUnboundVariability(Variability::Uniform);
}

// Private constructor for cloning.
NewExpr::NewExpr(const Type *type, Expr *count, Expr *init, bool isV, SourcePos p)
    : Expr(p, NewExprID), allocType(type), countExpr(count), initExpr(init), isVarying(isV) {}

llvm::Value *NewExpr::GetValue(FunctionEmitContext *ctx) const {
    bool do32Bit = (g->target->is32Bit() || g->opt.force32BitAddressing);

    // Determine how many elements we need to allocate.  Note that this
    // will be a varying value if this is a varying new.
    llvm::Value *countValue;
    if (countExpr != nullptr) {
        countValue = countExpr->GetValue(ctx);
        if (countValue == nullptr) {
            AssertPos(pos, m->errorCount > 0);
            return nullptr;
        }
    } else {
        if (isVarying) {
            if (do32Bit)
                countValue = LLVMInt32Vector(1);
            else
                countValue = LLVMInt64Vector(1);
        } else {
            if (do32Bit)
                countValue = LLVMInt32(1);
            else
                countValue = LLVMInt64(1);
        }
    }

    // Compute the total amount of memory to allocate, allocSize, as the
    // product of the number of elements to allocate and the size of a
    // single element.
    llvm::Type *llvmAllocType = allocType->LLVMType(g->ctx);
    Assert(llvmAllocType);
    llvm::Value *eltSize = g->target->SizeOf(llvmAllocType, ctx->GetCurrentBasicBlock());
    if (isVarying)
        eltSize = ctx->SmearUniform(eltSize, "smear_size");
    llvm::Value *allocSize =
        ctx->BinaryOperator(llvm::Instruction::Mul, countValue, eltSize, WrapSemantics::NSW, "alloc_size");

    // Determine which allocation builtin function to call: uniform or
    // varying, and taking 32-bit or 64-bit allocation counts.
    llvm::Function *func;
    if (isVarying) {
        if (g->target->is32Bit()) {
            func = m->module->getFunction(builtin::__new_varying32_32rt);
        } else if (g->opt.force32BitAddressing) {
            func = m->module->getFunction(builtin::__new_varying32_64rt);
        } else {
            func = m->module->getFunction(builtin::__new_varying64_64rt);
        }
    } else {
        // FIXME: __new_uniform_32rt should take i32
        if (allocSize->getType() != LLVMTypes::Int64Type)
            allocSize = ctx->SExtInst(allocSize, LLVMTypes::Int64Type, "alloc_size64");
        if (g->target->is32Bit()) {
            func = m->module->getFunction(builtin::__new_uniform_32rt);
        } else {
            func = m->module->getFunction(builtin::__new_uniform_64rt);
        }
    }
    AssertPos(pos, func != nullptr);

    // Make the call for the the actual allocation.
    llvm::Value *ptrValue = ctx->CallInst(func, nullptr, allocSize, "new");

    // Now handle initializers and returning the right type for the result.
    const Type *retType = GetType();
    if (retType == nullptr)
        return nullptr;
    if (isVarying) {
        if (g->target->is32Bit())
            // Convert i64 vector values to i32 if we are compiling to a
            // 32-bit target.
            ptrValue = ctx->TruncInst(ptrValue, LLVMTypes::VoidPointerVectorType, "ptr_to_32bit");

        if (initExpr != nullptr) {
            // If we have an initializer expression, emit code that checks
            // to see if each lane is active and if so, runs the code to do
            // the initialization.  Note that we're we're taking advantage
            // of the fact that the __new_varying*() functions are
            // implemented to return nullptr for program instances that aren't
            // executing; more generally, we should be using the current
            // execution mask for this...
            for (int i = 0; i < g->target->getVectorWidth(); ++i) {
                llvm::BasicBlock *bbInit = ctx->CreateBasicBlock("init_ptr");
                llvm::BasicBlock *bbSkip = ctx->CreateBasicBlock("skip_init");
                llvm::Value *p = ctx->ExtractInst(ptrValue, i);
                llvm::Value *nullValue = g->target->is32Bit() ? LLVMInt32(0) : LLVMInt64(0);
                // Is the pointer for the current lane non-zero?
                llvm::Value *nonNull =
                    ctx->CmpInst(llvm::Instruction::ICmp, llvm::CmpInst::ICMP_NE, p, nullValue, "non_null");
                ctx->BranchInst(bbInit, bbSkip, nonNull);

                // Initialize the memory pointed to by the pointer for the
                // current lane.
                ctx->SetCurrentBasicBlock(bbInit);
                llvm::Type *ptrType = retType->GetAsUniformType()->LLVMType(g->ctx);
                llvm::Value *ptr = ctx->IntToPtrInst(p, ptrType);
                InitSymbol(new AddressInfo(ptr, ptrType), allocType, initExpr, ctx, pos);
                ctx->BranchInst(bbSkip);

                ctx->SetCurrentBasicBlock(bbSkip);
            }
        }

        return ptrValue;
    } else {
        // For uniform news, we just need to cast the void * to be a
        // pointer of the return type and to run the code for initializers,
        // if present.
        llvm::Type *ptrType = retType->LLVMType(g->ctx);
        ptrValue = ctx->BitCastInst(ptrValue, ptrType, llvm::Twine(ptrValue->getName()) + "_cast_ptr");

        if (initExpr != nullptr)
            InitSymbol(new AddressInfo(ptrValue, ptrType), allocType, initExpr, ctx, pos);

        return ptrValue;
    }
}

const Type *NewExpr::GetType() const {
    if (allocType == nullptr)
        return nullptr;

    if (allocType->IsDependentType()) {
        return AtomicType::Dependent;
    }

    return isVarying ? PointerType::GetVarying(allocType) : PointerType::GetUniform(allocType);
}

Expr *NewExpr::TypeCheck() {
    // It's illegal to call new with an undefined struct type
    if (allocType == nullptr) {
        AssertPos(pos, m->errorCount > 0);
        return nullptr;
    }

    if (allocType->IsDependentType()) {
        return this;
    }

    if (g->target->isXeTarget()) {
        Error(pos, "\"new\" is not supported for Xe targets yet.");
        return nullptr;
    }

    if (CastType<UndefinedStructType>(allocType) != nullptr) {
        Error(pos,
              "Can't dynamically allocate storage for declared "
              "but not defined type \"%s\".",
              allocType->GetString().c_str());
        return nullptr;
    }
    const StructType *st = CastType<StructType>(allocType);
    if (st != nullptr && !st->IsDefined()) {
        Error(pos,
              "Can't dynamically allocate storage for declared "
              "type \"%s\" containing undefined member type.",
              allocType->GetString().c_str());
        return nullptr;
    }

    // Otherwise we only need to make sure that if we have an expression
    // giving a number of elements to allocate that it can be converted to
    // an integer of the appropriate variability.
    if (countExpr == nullptr)
        return this;

    const Type *countType;
    if ((countType = countExpr->GetType()) == nullptr)
        return nullptr;

    if (isVarying == false && countType->IsVaryingType()) {
        Error(pos, "Illegal to provide \"varying\" allocation count with "
                   "\"uniform new\" expression.");
        return nullptr;
    }

    // Figure out the type that the allocation count should be
    const Type *t =
        (g->target->is32Bit() || g->opt.force32BitAddressing) ? AtomicType::UniformUInt32 : AtomicType::UniformUInt64;
    if (isVarying)
        t = t->GetAsVaryingType();

    countExpr = TypeConvertExpr(countExpr, t, "item count");
    if (countExpr == nullptr)
        return nullptr;

    return this;
}

Expr *NewExpr::Optimize() { return this; }

void NewExpr::Print(Indent &indent) const {
    indent.Print("NewExpr", pos);

    printf("[%s] isVarying: %s\n", allocType ? allocType->GetString().c_str() : "<NULL allocType>",
           isVarying ? "true" : "false");

    if (countExpr || initExpr) {
        int kids = (countExpr ? 1 : 0) + (initExpr ? 1 : 0);
        indent.pushList(kids);
        if (countExpr) {
            indent.setNextLabel("count");
            countExpr->Print(indent);
        }
        if (initExpr) {
            indent.setNextLabel("init");
            initExpr->Print(indent);
        }
    }

    indent.Done();
}

int NewExpr::EstimateCost() const { return COST_NEW; }

NewExpr *NewExpr::Instantiate(TemplateInstantiation &templInst) const {
    const Type *instType = allocType ? allocType->ResolveDependenceForTopType(templInst) : nullptr;
    Expr *instInit = initExpr ? initExpr->Instantiate(templInst) : nullptr;
    Expr *instCount = countExpr ? countExpr->Instantiate(templInst) : nullptr;
    return new NewExpr(instType, instCount, instInit, isVarying, pos);
}
