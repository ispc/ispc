/*
  Copyright (c) 2025, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file constexpr.cpp
    @brief Helpers for constexpr validation and evaluation.
*/

#include "constexpr.h"

#include "ast.h"
#include "expr.h"
#include "func.h"
#include "llvmutil.h"
#include "stmt.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <llvm/IR/Constants.h>

using namespace ispc;

namespace {

constexpr int kMaxConstexprSteps = 100000;
constexpr int kMaxConstexprCallDepth = 64;

struct EvalResult {
    enum class Kind { Ok, Return, Break, Continue, Error } kind;
    ConstExpr *value;
};

static ConstExpr *lMakeZeroConstExpr(const Type *type, SourcePos pos);
static bool lIsNullPointerConstexpr(ConstExpr *value);
static llvm::Constant *lConvertConstexprPointer(llvm::Constant *c, const Type *toType);

static ConstExpr *lConvertConstExpr(ConstExpr *value, const Type *toType, SourcePos pos) {
    if (value == nullptr || toType == nullptr) {
        return nullptr;
    }
    if (Type::Equal(value->GetType(), toType)) {
        return value;
    }
    if (Type::EqualIgnoringConst(value->GetType(), toType)) {
        return value;
    }
    const Type *fromType = value->GetType();
    const Type *toBase = toType->GetAsNonConstType();
    const Type *fromBase = fromType ? fromType->GetAsNonConstType() : nullptr;
    const VectorType *toVector = CastType<VectorType>(toBase);
    const VectorType *fromVector = CastType<VectorType>(fromBase);
    const PointerType *toPointer = CastType<PointerType>(toBase);
    const PointerType *fromPointer = CastType<PointerType>(fromBase);

    if (value->IsAggregate() || toVector != nullptr || fromVector != nullptr) {
        if (toVector != nullptr && fromVector != nullptr) {
            if (toVector->GetElementCount() != fromVector->GetElementCount()) {
                return nullptr;
            }
            if (!value->IsAggregate()) {
                return nullptr;
            }
            std::vector<ConstExpr *> converted;
            const std::vector<ConstExpr *> &elements = value->GetAggregateValues();
            if ((int)elements.size() != toVector->GetElementCount()) {
                return nullptr;
            }
            converted.reserve(elements.size());
            for (ConstExpr *elem : elements) {
                ConstExpr *conv = lConvertConstExpr(elem, toVector->GetElementType(), pos);
                if (conv == nullptr) {
                    return nullptr;
                }
                converted.push_back(conv);
            }
            return new ConstExpr(toType, converted, pos);
        }
        if (toVector != nullptr && fromVector == nullptr) {
            if (value->IsAggregate()) {
                return nullptr;
            }
            ConstExpr *elem = lConvertConstExpr(value, toVector->GetElementType(), pos);
            if (elem == nullptr) {
                return nullptr;
            }
            std::vector<ConstExpr *> elements(toVector->GetElementCount(), elem);
            return new ConstExpr(toType, elements, pos);
        }
        if (toVector == nullptr && fromVector != nullptr) {
            if (!value->IsAggregate() || fromVector->GetElementCount() != 1) {
                return nullptr;
            }
            const std::vector<ConstExpr *> &elements = value->GetAggregateValues();
            if (elements.empty()) {
                return nullptr;
            }
            return lConvertConstExpr(elements[0], toType, pos);
        }
        return nullptr;
    }
    auto isZero = [](ConstExpr *candidate) -> bool {
        if (candidate == nullptr || candidate->IsAggregate()) {
            return false;
        }
        if (candidate->IsPointerConst()) {
            llvm::Constant *c = candidate->GetPointerConstant();
            return c != nullptr && c->isNullValue();
        }
        const Type *base = candidate->GetType() ? candidate->GetType()->GetAsNonConstType() : nullptr;
        const AtomicType *at = CastType<AtomicType>(base);
        const EnumType *et = CastType<EnumType>(base);
        const PointerType *pt = CastType<PointerType>(base);
        if (pt == nullptr && et == nullptr && (at == nullptr || (!at->IsIntType() && !at->IsBoolType()))) {
            return false;
        }
        int64_t vals[ISPC_MAX_NVEC] = {};
        candidate->GetValues(vals, candidate->GetType()->IsVaryingType());
        for (int i = 0; i < candidate->Count(); ++i) {
            if (vals[i] != 0) {
                return false;
            }
        }
        return true;
    };

    if (toPointer != nullptr || fromPointer != nullptr) {
        if (value->GetType() && value->GetType()->IsVaryingType() && toType->IsUniformType()) {
            return nullptr;
        }
        if (toPointer != nullptr && fromPointer != nullptr) {
            if (value->IsPointerConst()) {
                llvm::Constant *c = lConvertConstexprPointer(value->GetPointerConstant(), toType);
                if (c == nullptr) {
                    return nullptr;
                }
                return new ConstExpr(toType, c, pos);
            }
            if (isZero(value)) {
                return lMakeZeroConstExpr(toType, pos);
            }
            return nullptr;
        }
        if (fromPointer != nullptr && toPointer == nullptr) {
            const AtomicType *toAtomic = CastType<AtomicType>(toBase);
            if (toAtomic != nullptr && toAtomic->IsBoolType()) {
                bool isNull = lIsNullPointerConstexpr(value) || isZero(value);
                if (toType->IsVaryingType()) {
                    bool vals[ISPC_MAX_NVEC];
                    for (int i = 0; i < g->target->getVectorWidth(); ++i) {
                        vals[i] = !isNull;
                    }
                    return new ConstExpr(toType->GetAsConstType(), vals, pos);
                }
                return new ConstExpr(toType->GetAsConstType(), !isNull, pos);
            }
            return nullptr;
        }
        if (toPointer != nullptr && fromPointer == nullptr) {
            if (isZero(value)) {
                return lMakeZeroConstExpr(toType, pos);
            }
            return nullptr;
        }
    }

    Expr *converted = TypeConvertExpr(value, toType, "constexpr evaluation");
    if (converted == nullptr) {
        return nullptr;
    }
    converted = TypeCheckAndOptimize(converted);
    return llvm::dyn_cast<ConstExpr>(converted);
}

static bool lIsNullPointerConstexpr(ConstExpr *value) {
    if (value == nullptr || value->IsAggregate()) {
        return false;
    }
    if (value->IsPointerConst()) {
        llvm::Constant *c = value->GetPointerConstant();
        return c != nullptr && c->isNullValue();
    }
    const Type *base = value->GetType() ? value->GetType()->GetAsNonConstType() : nullptr;
    if (CastType<PointerType>(base) == nullptr) {
        return false;
    }
    int64_t vals[ISPC_MAX_NVEC] = {};
    value->GetValues(vals, value->GetType()->IsVaryingType());
    for (int i = 0; i < value->Count(); ++i) {
        if (vals[i] != 0) {
            return false;
        }
    }
    return true;
}

static llvm::Constant *lConvertConstexprPointer(llvm::Constant *c, const Type *toType) {
    if (c == nullptr || toType == nullptr) {
        return nullptr;
    }
    llvm::Type *llvmType = toType->LLVMType(g->ctx);
    if (llvmType == nullptr) {
        return nullptr;
    }
    if (c->getType() == llvmType) {
        return c;
    }
    if (llvmType->isPointerTy()) {
        if (c->getType()->isPointerTy()) {
            return llvm::ConstantExpr::getBitCast(c, llvmType);
        }
        return llvm::ConstantExpr::getIntToPtr(c, llvmType);
    }
    llvm::Constant *intPtr =
        c->getType()->isPointerTy() ? llvm::ConstantExpr::getPtrToInt(c, LLVMTypes::PointerIntType) : c;
    int count = toType->IsVaryingType() ? g->target->getVectorWidth() : toType->GetSOAWidth();
    std::vector<llvm::Constant *> smear;
    smear.reserve(count);
    for (int i = 0; i < count; ++i) {
        smear.push_back(intPtr);
    }
    if (toType->IsVaryingType()) {
        return llvm::ConstantVector::get(smear);
    }
    llvm::ArrayType *at = llvm::ArrayType::get(LLVMTypes::PointerIntType, count);
    return llvm::ConstantArray::get(at, smear);
}

static ConstExpr *lConstExprFromConstant(const Type *type, llvm::Constant *constant, SourcePos pos) {
    if (type == nullptr || constant == nullptr) {
        return nullptr;
    }
    const Type *baseType = type->GetAsNonConstType();
    if (baseType->IsVaryingType()) {
        baseType = baseType->GetAsUniformType();
    }
    const AtomicType *at = CastType<AtomicType>(baseType);
    const EnumType *et = CastType<EnumType>(baseType);
    if (at == nullptr && et == nullptr) {
        return nullptr;
    }
    if (llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(constant)) {
        int64_t v = ci->getSExtValue();
        switch (at ? at->basicType : AtomicType::TYPE_UINT32) {
        case AtomicType::TYPE_BOOL:
            return new ConstExpr(type->GetAsConstType(), (bool)(v != 0), pos);
        case AtomicType::TYPE_INT8:
            return new ConstExpr(type->GetAsConstType(), (int8_t)v, pos);
        case AtomicType::TYPE_UINT8:
            return new ConstExpr(type->GetAsConstType(), (uint8_t)v, pos);
        case AtomicType::TYPE_INT16:
            return new ConstExpr(type->GetAsConstType(), (int16_t)v, pos);
        case AtomicType::TYPE_UINT16:
            return new ConstExpr(type->GetAsConstType(), (uint16_t)v, pos);
        case AtomicType::TYPE_INT32:
            return new ConstExpr(type->GetAsConstType(), (int32_t)v, pos);
        case AtomicType::TYPE_UINT32:
            return new ConstExpr(type->GetAsConstType(), (uint32_t)ci->getZExtValue(), pos);
        case AtomicType::TYPE_INT64:
            return new ConstExpr(type->GetAsConstType(), (int64_t)v, pos);
        case AtomicType::TYPE_UINT64:
            return new ConstExpr(type->GetAsConstType(), (uint64_t)ci->getZExtValue(), pos);
        default:
            return nullptr;
        }
    }
    if (llvm::ConstantFP *cf = llvm::dyn_cast<llvm::ConstantFP>(constant)) {
        llvm::APFloat val = cf->getValueAPF();
        switch (at ? at->basicType : AtomicType::TYPE_UINT32) {
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
        case AtomicType::TYPE_DOUBLE:
            return new ConstExpr(type->GetAsConstType(), val, pos);
        default:
            return nullptr;
        }
    }
    return nullptr;
}

static int lVectorSwizzleIndex(char id) {
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

static ConstExpr *lMakeZeroConstExpr(const Type *type, SourcePos pos) {
    if (type == nullptr) {
        return nullptr;
    }
    const Type *baseType = type->GetAsNonConstType();
    if (const AtomicType *at = CastType<AtomicType>(baseType)) {
        const Type *constType = type->GetAsConstType();
        switch (at->basicType) {
        case AtomicType::TYPE_BOOL:
            if (type->IsVaryingType()) {
                bool vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, false, pos);
        case AtomicType::TYPE_INT8: {
            if (type->IsVaryingType()) {
                int8_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (int8_t)0, pos);
        }
        case AtomicType::TYPE_UINT8: {
            if (type->IsVaryingType()) {
                uint8_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (uint8_t)0, pos);
        }
        case AtomicType::TYPE_INT16: {
            if (type->IsVaryingType()) {
                int16_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (int16_t)0, pos);
        }
        case AtomicType::TYPE_UINT16: {
            if (type->IsVaryingType()) {
                uint16_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (uint16_t)0, pos);
        }
        case AtomicType::TYPE_INT32: {
            if (type->IsVaryingType()) {
                int32_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (int32_t)0, pos);
        }
        case AtomicType::TYPE_UINT32: {
            if (type->IsVaryingType()) {
                uint32_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (uint32_t)0, pos);
        }
        case AtomicType::TYPE_INT64: {
            if (type->IsVaryingType()) {
                int64_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (int64_t)0, pos);
        }
        case AtomicType::TYPE_UINT64: {
            if (type->IsVaryingType()) {
                uint64_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, (uint64_t)0, pos);
        }
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
        case AtomicType::TYPE_DOUBLE: {
            llvm::Type *llvmType = type->GetAsUniformType()->LLVMType(g->ctx);
            if (llvmType == nullptr) {
                return nullptr;
            }
            llvm::APFloat zero = llvm::APFloat::getZero(llvmType->getFltSemantics());
            if (type->IsVaryingType()) {
                std::vector<llvm::APFloat> vals;
                vals.resize(g->target->getVectorWidth(), zero);
                return new ConstExpr(constType, vals, pos);
            }
            return new ConstExpr(constType, zero, pos);
        }
        default:
            return nullptr;
        }
    }
    if (CastType<EnumType>(baseType) != nullptr) {
        const Type *constType = type->GetAsConstType();
        if (type->IsVaryingType()) {
            uint32_t vals[ISPC_MAX_NVEC] = {};
            return new ConstExpr(constType, vals, pos);
        }
        return new ConstExpr(constType, (uint32_t)0, pos);
    }
    if (CastType<PointerType>(baseType) != nullptr) {
        const Type *constType = type->GetAsConstType();
        if (type->IsVaryingType()) {
            if (g->target->is32Bit()) {
                uint32_t vals[ISPC_MAX_NVEC] = {};
                return new ConstExpr(constType, vals, pos);
            }
            uint64_t vals[ISPC_MAX_NVEC] = {};
            return new ConstExpr(constType, vals, pos);
        }
        if (g->target->is32Bit()) {
            return new ConstExpr(constType, (uint32_t)0, pos);
        }
        return new ConstExpr(constType, (uint64_t)0, pos);
    }
    if (const CollectionType *ct = CastType<CollectionType>(baseType)) {
        int count = ct->GetElementCount();
        std::vector<ConstExpr *> elements;
        elements.reserve(count);
        for (int i = 0; i < count; ++i) {
            ConstExpr *elem = lMakeZeroConstExpr(ct->GetElementType(i), pos);
            if (elem == nullptr) {
                return nullptr;
            }
            elements.push_back(elem);
        }
        return new ConstExpr(type->GetAsConstType(), elements, pos);
    }
    return nullptr;
}

class ConstexprEvaluator {
  public:
    explicit ConstexprEvaluator(int callDepth = 0, bool allowDeferredEval = false)
        : depth(callDepth), steps(0), hasError(false), allowDeferred(allowDeferredEval), deferred(false),
          targetDependent(false) {}

    ConstExpr *EvalExpr(const Expr *expr);
    ConstExpr *EvalExprList(const ExprList *exprList, const Type *expectedType);
    ConstExpr *EvalFunction(const Function *func, const std::vector<ConstExpr *> &args);
    bool Failed() const { return hasError; }
    bool Deferred() const { return deferred; }
    bool IsTargetDependent() const { return targetDependent; }

  private:
    struct Scope {
        std::unordered_map<const Symbol *, ConstExpr *> values;
    };

    std::vector<Scope> scopes;
    int depth;
    int steps;
    bool hasError;
    bool allowDeferred;
    bool deferred;
    bool targetDependent;

    void SetError() { hasError = true; }
    void SetDeferred() { deferred = true; }
    void SetTargetDependent() { targetDependent = true; }

    void PushScope() { scopes.push_back(Scope()); }
    void PopScope() {
        if (!scopes.empty()) {
            scopes.pop_back();
        }
    }

    void DeclareValue(const Symbol *sym, ConstExpr *value) {
        if (scopes.empty()) {
            PushScope();
        }
        scopes.back().values[sym] = value;
    }

    bool UpdateValue(const Symbol *sym, ConstExpr *value) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->values.find(sym);
            if (found != it->values.end()) {
                found->second = value;
                return true;
            }
        }
        return false;
    }

    ConstExpr *LookupValue(const Symbol *sym) const {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->values.find(sym);
            if (found != it->values.end()) {
                return found->second;
            }
        }
        return nullptr;
    }

    EvalResult EvalStmt(const Stmt *stmt);
    EvalResult EvalStmtList(const StmtList *stmtList);

    ConstExpr *FoldExpr(Expr *expr) {
        Expr *folded = TypeCheckAndOptimize(expr);
        return llvm::dyn_cast<ConstExpr>(folded);
    }

    ConstExpr *FoldUnary(UnaryExpr::Op op, ConstExpr *arg, SourcePos pos) {
        if (arg == nullptr) {
            return nullptr;
        }
        const Type *argBase = arg->GetType() ? arg->GetType()->GetAsNonConstType() : nullptr;
        if (CastType<PointerType>(argBase) != nullptr && op == UnaryExpr::LogicalNot) {
            bool isNull = lIsNullPointerConstexpr(arg);
            if (arg->GetType()->IsVaryingType()) {
                bool vals[ISPC_MAX_NVEC];
                for (int i = 0; i < g->target->getVectorWidth(); ++i) {
                    vals[i] = isNull;
                }
                return new ConstExpr(AtomicType::VaryingBool->GetAsConstType(), vals, pos);
            }
            return new ConstExpr(AtomicType::UniformBool->GetAsConstType(), isNull, pos);
        }
        if (arg->IsAggregate()) {
            const VectorType *vt = CastType<VectorType>(arg->GetType()->GetAsNonConstType());
            if (vt == nullptr) {
                return nullptr;
            }
            const std::vector<ConstExpr *> &elements = arg->GetAggregateValues();
            if ((int)elements.size() != vt->GetElementCount()) {
                return nullptr;
            }
            std::vector<ConstExpr *> results;
            results.reserve(elements.size());
            const Type *elementType = nullptr;
            for (ConstExpr *elem : elements) {
                ConstExpr *res = FoldUnary(op, elem, pos);
                if (res == nullptr) {
                    return nullptr;
                }
                if (elementType == nullptr) {
                    elementType = res->GetType();
                } else if (!Type::EqualIgnoringConst(elementType, res->GetType())) {
                    return nullptr;
                }
                results.push_back(res);
            }
            if (elementType == nullptr) {
                return nullptr;
            }
            const VectorType *resultType = new VectorType(elementType, vt->GetElementCount());
            return new ConstExpr(resultType, results, pos);
        }
        UnaryExpr *expr = new UnaryExpr(op, arg, pos);
        return FoldExpr(expr);
    }

    ConstExpr *FoldBinary(BinaryExpr::Op op, ConstExpr *lhs, ConstExpr *rhs, SourcePos pos) {
        if (lhs == nullptr || rhs == nullptr) {
            return nullptr;
        }
        const VectorType *lhsVec = lhs->IsAggregate() ? CastType<VectorType>(lhs->GetType()->GetAsNonConstType())
                                                      : nullptr;
        const VectorType *rhsVec = rhs->IsAggregate() ? CastType<VectorType>(rhs->GetType()->GetAsNonConstType())
                                                      : nullptr;
        if (!lhs->IsAggregate() && !rhs->IsAggregate()) {
            BinaryExpr *expr = new BinaryExpr(op, lhs, rhs, pos);
            return FoldExpr(expr);
        }
        if ((lhs->IsAggregate() && lhsVec == nullptr) || (rhs->IsAggregate() && rhsVec == nullptr)) {
            return nullptr;
        }
        if (lhsVec != nullptr && rhsVec == nullptr) {
            ConstExpr *elem = lConvertConstExpr(rhs, lhsVec->GetElementType(), pos);
            if (elem == nullptr) {
                return nullptr;
            }
            std::vector<ConstExpr *> elems(lhsVec->GetElementCount(), elem);
            rhs = new ConstExpr(lhs->GetType(), elems, pos);
            rhsVec = lhsVec;
        } else if (lhsVec == nullptr && rhsVec != nullptr) {
            ConstExpr *elem = lConvertConstExpr(lhs, rhsVec->GetElementType(), pos);
            if (elem == nullptr) {
                return nullptr;
            }
            std::vector<ConstExpr *> elems(rhsVec->GetElementCount(), elem);
            lhs = new ConstExpr(rhs->GetType(), elems, pos);
            lhsVec = rhsVec;
        }
        if (lhsVec == nullptr || rhsVec == nullptr) {
            return nullptr;
        }
        if (lhsVec->GetElementCount() != rhsVec->GetElementCount()) {
            return nullptr;
        }
        const std::vector<ConstExpr *> &lhsElems = lhs->GetAggregateValues();
        const std::vector<ConstExpr *> &rhsElems = rhs->GetAggregateValues();
        if (lhsElems.size() != rhsElems.size()) {
            return nullptr;
        }
        std::vector<ConstExpr *> results;
        results.reserve(lhsElems.size());
        const Type *elementType = nullptr;
        for (size_t i = 0; i < lhsElems.size(); ++i) {
            ConstExpr *res = FoldBinary(op, lhsElems[i], rhsElems[i], pos);
            if (res == nullptr) {
                return nullptr;
            }
            if (elementType == nullptr) {
                elementType = res->GetType();
            } else if (!Type::EqualIgnoringConst(elementType, res->GetType())) {
                return nullptr;
            }
            results.push_back(res);
        }
        if (elementType == nullptr) {
            return nullptr;
        }
        const VectorType *resultType = new VectorType(elementType, lhsVec->GetElementCount());
        return new ConstExpr(resultType, results, pos);
    }

    ConstExpr *FoldSelect(ConstExpr *cond, ConstExpr *tExpr, ConstExpr *fExpr, SourcePos pos) {
        if (cond == nullptr || tExpr == nullptr || fExpr == nullptr) {
            return nullptr;
        }
        if (!cond->IsAggregate()) {
            const Type *condBase = cond->GetType() ? cond->GetType()->GetAsNonConstType() : nullptr;
            if (CastType<PointerType>(condBase) != nullptr) {
                ConstExpr *condBool = lConvertConstExpr(cond, AtomicType::UniformBool, pos);
                if (condBool == nullptr) {
                    return nullptr;
                }
                cond = condBool;
            }
        }
        if (tExpr->IsAggregate() || fExpr->IsAggregate()) {
            const VectorType *tVec = tExpr->IsAggregate() ? CastType<VectorType>(tExpr->GetType()->GetAsNonConstType())
                                                          : nullptr;
            const VectorType *fVec = fExpr->IsAggregate() ? CastType<VectorType>(fExpr->GetType()->GetAsNonConstType())
                                                          : nullptr;
            if (tVec == nullptr || fVec == nullptr || tVec->GetElementCount() != fVec->GetElementCount()) {
                return nullptr;
            }
            const std::vector<ConstExpr *> &tElems = tExpr->GetAggregateValues();
            const std::vector<ConstExpr *> &fElems = fExpr->GetAggregateValues();
            if (tElems.size() != fElems.size()) {
                return nullptr;
            }
            std::vector<ConstExpr *> results;
            results.reserve(tElems.size());
            if (!cond->IsAggregate() && cond->Count() == 1) {
                bool cv[ISPC_MAX_NVEC];
                int count = cond->GetValues(cv);
                if (count != 1) {
                    return nullptr;
                }
                return cv[0] ? tExpr : fExpr;
            }
            if (cond->IsAggregate()) {
                const std::vector<ConstExpr *> &cElems = cond->GetAggregateValues();
                if (cElems.size() != tElems.size()) {
                    return nullptr;
                }
                for (size_t i = 0; i < tElems.size(); ++i) {
                    ConstExpr *res = FoldSelect(cElems[i], tElems[i], fElems[i], pos);
                    if (res == nullptr) {
                        return nullptr;
                    }
                    results.push_back(res);
                }
            } else {
                for (size_t i = 0; i < tElems.size(); ++i) {
                    ConstExpr *res = FoldSelect(cond, tElems[i], fElems[i], pos);
                    if (res == nullptr) {
                        return nullptr;
                    }
                    results.push_back(res);
                }
            }
            return new ConstExpr(tExpr->GetType(), results, pos);
        }
        SelectExpr *expr = new SelectExpr(cond, tExpr, fExpr, pos);
        return FoldExpr(expr);
    }

    ConstExpr *FoldCast(const Type *toType, ConstExpr *arg, SourcePos pos) {
        if (arg == nullptr) {
            return nullptr;
        }
        const Type *toBase = toType->GetAsNonConstType();
        if (arg->IsAggregate() || CastType<CollectionType>(toBase) != nullptr ||
            CastType<PointerType>(toBase) != nullptr) {
            return lConvertConstExpr(arg, toType, pos);
        }
        TypeCastExpr *expr = new TypeCastExpr(toType, arg, pos);
        return FoldExpr(expr);
    }

    ConstExpr *EvalAssign(const AssignExpr *expr);
    ConstExpr *EvalIncDec(const UnaryExpr *expr);
};

ConstExpr *ConstexprEvaluator::EvalAssign(const AssignExpr *expr) {
    const SymbolExpr *se = llvm::dyn_cast<SymbolExpr>(expr->lvalue);
    if (se == nullptr) {
        SetError();
        return nullptr;
    }
    Symbol *sym = se->GetBaseSymbol();
    if (sym == nullptr || sym->type == nullptr || sym->type->IsConstType()) {
        SetError();
        return nullptr;
    }
    ConstExpr *rhs = EvalExpr(expr->rvalue);
    if (rhs == nullptr) {
        SetError();
        return nullptr;
    }
    if (expr->op != AssignExpr::Assign) {
        ConstExpr *lhs = LookupValue(sym);
        if (lhs == nullptr) {
            SetError();
            return nullptr;
        }
        BinaryExpr::Op bop = BinaryExpr::Add;
        switch (expr->op) {
        case AssignExpr::MulAssign:
            bop = BinaryExpr::Mul;
            break;
        case AssignExpr::DivAssign:
            bop = BinaryExpr::Div;
            break;
        case AssignExpr::ModAssign:
            bop = BinaryExpr::Mod;
            break;
        case AssignExpr::AddAssign:
            bop = BinaryExpr::Add;
            break;
        case AssignExpr::SubAssign:
            bop = BinaryExpr::Sub;
            break;
        case AssignExpr::ShlAssign:
            bop = BinaryExpr::Shl;
            break;
        case AssignExpr::ShrAssign:
            bop = BinaryExpr::Shr;
            break;
        case AssignExpr::AndAssign:
            bop = BinaryExpr::BitAnd;
            break;
        case AssignExpr::XorAssign:
            bop = BinaryExpr::BitXor;
            break;
        case AssignExpr::OrAssign:
            bop = BinaryExpr::BitOr;
            break;
        default:
            SetError();
            return nullptr;
        }
        rhs = FoldBinary(bop, lhs, rhs, expr->pos);
        if (rhs == nullptr) {
            SetError();
            return nullptr;
        }
    }
    if (!UpdateValue(sym, rhs)) {
        SetError();
        return nullptr;
    }
    return rhs;
}

ConstExpr *ConstexprEvaluator::EvalIncDec(const UnaryExpr *expr) {
    const SymbolExpr *se = llvm::dyn_cast<SymbolExpr>(expr->expr);
    if (se == nullptr) {
        SetError();
        return nullptr;
    }
    Symbol *sym = se->GetBaseSymbol();
    if (sym == nullptr || sym->type == nullptr || sym->type->IsConstType()) {
        SetError();
        return nullptr;
    }
    ConstExpr *current = LookupValue(sym);
    if (current == nullptr) {
        SetError();
        return nullptr;
    }
    ConstExpr *one = new ConstExpr(AtomicType::UniformInt32->GetAsConstType(), 1, expr->pos);
    BinaryExpr::Op op = (expr->op == UnaryExpr::PreDec || expr->op == UnaryExpr::PostDec) ? BinaryExpr::Sub
                                                                                           : BinaryExpr::Add;
    ConstExpr *updated = FoldBinary(op, current, one, expr->pos);
    if (updated == nullptr) {
        SetError();
        return nullptr;
    }
    if (!UpdateValue(sym, updated)) {
        SetError();
        return nullptr;
    }
    if (expr->op == UnaryExpr::PostInc || expr->op == UnaryExpr::PostDec) {
        return current;
    }
    return updated;
}

ConstExpr *ConstexprEvaluator::EvalExpr(const Expr *expr) {
    if (expr == nullptr || hasError) {
        return nullptr;
    }
    if (auto *ce = llvm::dyn_cast<ConstExpr>(expr)) {
        return const_cast<ConstExpr *>(ce);
    }
    if (auto *npe = llvm::dyn_cast<NullPointerExpr>(expr)) {
        return lMakeZeroConstExpr(npe->GetType(), npe->pos);
    }
    if (auto *se = llvm::dyn_cast<SymbolExpr>(expr)) {
        Symbol *sym = se->GetBaseSymbol();
        if (sym != nullptr) {
            if (ConstExpr *local = LookupValue(sym)) {
                return local;
            }
            if (sym->constValue) {
                return sym->constValue;
            }
            if (allowDeferred && sym->constEvalPending) {
                SetDeferred();
                return nullptr;
            }
        }
        SetError();
        return nullptr;
    }
    if (auto *ue = llvm::dyn_cast<UnaryExpr>(expr)) {
        if (ue->op == UnaryExpr::PreInc || ue->op == UnaryExpr::PreDec ||
            ue->op == UnaryExpr::PostInc || ue->op == UnaryExpr::PostDec) {
            return EvalIncDec(ue);
        }
        ConstExpr *arg = EvalExpr(ue->expr);
        if (arg == nullptr) {
            SetError();
            return nullptr;
        }
        ConstExpr *result = FoldUnary(ue->op, arg, ue->pos);
        if (result == nullptr) {
            SetError();
        }
        return result;
    }
    if (auto *be = llvm::dyn_cast<BinaryExpr>(expr)) {
        if (be->op == BinaryExpr::Comma) {
            (void)EvalExpr(be->arg0);
            return EvalExpr(be->arg1);
        }
        ConstExpr *lhs = EvalExpr(be->arg0);
        ConstExpr *rhs = EvalExpr(be->arg1);
        if (lhs == nullptr || rhs == nullptr) {
            SetError();
            return nullptr;
        }
        const Type *lhsBase = lhs->GetType() ? lhs->GetType()->GetAsNonConstType() : nullptr;
        const Type *rhsBase = rhs->GetType() ? rhs->GetType()->GetAsNonConstType() : nullptr;
        const PointerType *lhsPtr = CastType<PointerType>(lhsBase);
        const PointerType *rhsPtr = CastType<PointerType>(rhsBase);
        if (lhsPtr != nullptr || rhsPtr != nullptr) {
            if (be->op == BinaryExpr::Equal || be->op == BinaryExpr::NotEqual) {
                bool lhsNull = lIsNullPointerConstexpr(lhs);
                bool rhsNull = lIsNullPointerConstexpr(rhs);
                bool equal = false;
                if (lhsNull || rhsNull) {
                    equal = lhsNull && rhsNull;
                } else if (lhs->IsPointerConst() && rhs->IsPointerConst()) {
                    equal = lhs->GetPointerConstant() == rhs->GetPointerConstant();
                } else {
                    SetError();
                    return nullptr;
                }
                bool result = (be->op == BinaryExpr::Equal) ? equal : !equal;
                return new ConstExpr(AtomicType::UniformBool->GetAsConstType(), result, be->pos);
            }
            if (be->op == BinaryExpr::Add || be->op == BinaryExpr::Sub) {
                if (lhsPtr != nullptr && rhsPtr != nullptr) {
                    Error(be->pos, "constexpr pointer arithmetic requires a pointer and integer offset.");
                    SetError();
                    return nullptr;
                }
                const Type *ptrType = lhsPtr ? lhs->GetType() : rhs->GetType();
                if (ptrType == nullptr || ptrType->IsVaryingType()) {
                    Error(be->pos, "constexpr pointer arithmetic requires a uniform pointer.");
                    SetError();
                    return nullptr;
                }
                BinaryExpr *tmp = new BinaryExpr(be->op, lhs, rhs, be->pos);
                Expr *typed = TypeCheckAndOptimize(tmp);
                if (typed == nullptr || typed->GetType() == nullptr) {
                    SetError();
                    return nullptr;
                }
                std::pair<llvm::Constant *, bool> cpair = typed->GetConstant(typed->GetType());
                if (cpair.first == nullptr) {
                    SetError();
                    return nullptr;
                }
                if (cpair.second) {
                    SetTargetDependent();
                }
                return new ConstExpr(typed->GetType(), cpair.first, be->pos);
            }
            SetError();
            return nullptr;
        }
        ConstExpr *result = FoldBinary(be->op, lhs, rhs, be->pos);
        if (result == nullptr) {
            SetError();
        }
        return result;
    }
    if (auto *ae = llvm::dyn_cast<AssignExpr>(expr)) {
        return EvalAssign(ae);
    }
    if (auto *te = llvm::dyn_cast<TypeCastExpr>(expr)) {
        ConstExpr *arg = EvalExpr(te->expr);
        if (arg == nullptr) {
            SetError();
            return nullptr;
        }
        ConstExpr *result = FoldCast(te->GetType(), arg, te->pos);
        if (result == nullptr) {
            SetError();
        }
        return result;
    }
    if (auto *se = llvm::dyn_cast<SelectExpr>(expr)) {
        ConstExpr *cond = EvalExpr(se->test);
        ConstExpr *tExpr = EvalExpr(se->expr1);
        ConstExpr *fExpr = EvalExpr(se->expr2);
        if (cond == nullptr || tExpr == nullptr || fExpr == nullptr) {
            SetError();
            return nullptr;
        }
        ConstExpr *result = FoldSelect(cond, tExpr, fExpr, se->pos);
        if (result == nullptr) {
            SetError();
        }
        return result;
    }
    if (auto *fce = llvm::dyn_cast<FunctionCallExpr>(expr)) {
        if (depth >= kMaxConstexprCallDepth) {
            SetError();
            return nullptr;
        }
        const FunctionSymbolExpr *fse = llvm::dyn_cast<FunctionSymbolExpr>(fce->func);
        if (fse == nullptr) {
            SetError();
            return nullptr;
        }
        Symbol *calleeSym = fse->GetBaseSymbol();
        if (calleeSym == nullptr) {
            SetError();
            return nullptr;
        }
        const FunctionType *calleeType = CastType<FunctionType>(calleeSym->type);
        if (calleeType == nullptr || !calleeType->IsConstexpr()) {
            SetError();
            return nullptr;
        }
        const Function *calleeFunc = calleeSym->parentFunction;
        if (calleeFunc == nullptr || calleeFunc->GetCode() == nullptr) {
            if (allowDeferred) {
                SetDeferred();
                return nullptr;
            }
            SetError();
            return nullptr;
        }

        std::vector<ConstExpr *> argValues;
        int provided = fce->args ? (int)fce->args->exprs.size() : 0;
        for (int i = 0; i < provided; ++i) {
            ConstExpr *val = EvalExpr(fce->args->exprs[i]);
            if (val == nullptr) {
                SetError();
                return nullptr;
            }
            val = lConvertConstExpr(val, calleeType->GetParameterType(i), fce->args->exprs[i]->pos);
            if (val == nullptr) {
                SetError();
                return nullptr;
            }
            argValues.push_back(val);
        }
        for (int i = provided; i < calleeType->GetNumParameters(); ++i) {
            Expr *def = calleeType->GetParameterDefault(i);
            if (def == nullptr) {
                SetError();
                return nullptr;
            }
            ConstExpr *val = EvalExpr(def);
            if (val == nullptr) {
                SetError();
                return nullptr;
            }
            val = lConvertConstExpr(val, calleeType->GetParameterType(i), def->pos);
            if (val == nullptr) {
                SetError();
                return nullptr;
            }
            argValues.push_back(val);
        }

        ConstexprEvaluator nested(depth + 1, allowDeferred);
        ConstExpr *result = nested.EvalFunction(calleeFunc, argValues);
        if (nested.Deferred()) {
            SetDeferred();
            return nullptr;
        }
        if (result == nullptr || nested.Failed()) {
            SetError();
            return nullptr;
        }
        if (nested.IsTargetDependent()) {
            SetTargetDependent();
        }
        return result;
    }
    if (auto *ao = llvm::dyn_cast<AddressOfExpr>(expr)) {
        const Type *ptrType = ao->GetType();
        if (ptrType == nullptr) {
            SetError();
            return nullptr;
        }
        std::pair<llvm::Constant *, bool> cpair = ao->GetConstant(ptrType);
        if (cpair.first == nullptr) {
            SetError();
            return nullptr;
        }
        if (cpair.second) {
            SetTargetDependent();
        }
        return new ConstExpr(ptrType, cpair.first, ao->pos);
    }
    if (auto *ie = llvm::dyn_cast<IndexExpr>(expr)) {
        ConstExpr *base = EvalExpr(ie->baseExpr);
        ConstExpr *idxExpr = EvalExpr(ie->index);
        if (base == nullptr || idxExpr == nullptr || !base->IsAggregate()) {
            SetError();
            return nullptr;
        }
        ConstExpr *idxConst = lConvertConstExpr(idxExpr, AtomicType::UniformInt64, ie->index->pos);
        if (idxConst == nullptr) {
            SetError();
            return nullptr;
        }
        int64_t idxVals[ISPC_MAX_NVEC];
        int count = idxConst->GetValues(idxVals);
        if (count != 1) {
            Error(ie->pos, "constexpr array/vector index must be a uniform constant.");
            SetError();
            return nullptr;
        }
        int64_t idx = idxVals[0];
        const std::vector<ConstExpr *> &elements = base->GetAggregateValues();
        if (idx < 0 || idx >= (int64_t)elements.size()) {
            Error(ie->pos, "constexpr array/vector index is out of bounds.");
            SetError();
            return nullptr;
        }
        return elements[(int)idx];
    }
    if (auto *me = llvm::dyn_cast<MemberExpr>(expr)) {
        ConstExpr *base = EvalExpr(me->expr);
        if (base == nullptr || !base->IsAggregate()) {
            SetError();
            return nullptr;
        }
        const std::vector<ConstExpr *> &elements = base->GetAggregateValues();
        if (auto *sme = llvm::dyn_cast<StructMemberExpr>(me)) {
            int idx = sme->getElementNumber();
            if (idx < 0 || idx >= (int)elements.size()) {
                SetError();
                return nullptr;
            }
            return elements[idx];
        }
        if (auto *vme = llvm::dyn_cast<VectorMemberExpr>(me)) {
            const std::string &swizzle = me->identifier;
            if (swizzle.size() == 1) {
                int idx = vme->getElementNumber();
                if (idx < 0 || idx >= (int)elements.size()) {
                    SetError();
                    return nullptr;
                }
                return elements[idx];
            }
            std::vector<ConstExpr *> swizzled;
            swizzled.reserve(swizzle.size());
            for (char ch : swizzle) {
                int idx = lVectorSwizzleIndex(ch);
                if (idx < 0 || idx >= (int)elements.size()) {
                    Error(me->pos, "Invalid swizzle character '%c' for constexpr vector.", ch);
                    SetError();
                    return nullptr;
                }
                swizzled.push_back(elements[idx]);
            }
            return new ConstExpr(vme->getElementType(), swizzled, me->pos);
        }
        SetError();
        return nullptr;
    }
    if (auto *sizeofExpr = llvm::dyn_cast<SizeOfExpr>(expr)) {
        std::pair<llvm::Constant *, bool> cpair = sizeofExpr->GetConstant(sizeofExpr->GetType());
        ConstExpr *value = lConstExprFromConstant(sizeofExpr->GetType(), cpair.first, sizeofExpr->pos);
        if (value == nullptr) {
            SetError();
        }
        if (cpair.second) {
            SetTargetDependent();
        }
        return value;
    }

    SetError();
    return nullptr;
}

ConstExpr *ConstexprEvaluator::EvalExprList(const ExprList *exprList, const Type *expectedType) {
    if (exprList == nullptr || expectedType == nullptr || hasError) {
        SetError();
        return nullptr;
    }

    const Type *baseType = expectedType->GetAsNonConstType();
    const AtomicType *atomic = CastType<AtomicType>(baseType);
    const EnumType *enumType = CastType<EnumType>(baseType);
    const CollectionType *collection = CastType<CollectionType>(baseType);
    const int nInits = (int)exprList->exprs.size();

    auto evalElement = [&](Expr *expr, const Type *elementType) -> ConstExpr * {
        if (expr == nullptr || llvm::dyn_cast<ExprList>(expr) != nullptr) {
            return nullptr;
        }
        Expr *checked = TypeCheckAndOptimize(expr);
        if (checked == nullptr) {
            return nullptr;
        }
        Expr *converted = TypeConvertExpr(checked, elementType, "initializer list");
        if (converted == nullptr) {
            return nullptr;
        }
        converted = TypeCheckAndOptimize(converted);
        if (converted == nullptr) {
            return nullptr;
        }
        ConstExpr *val = EvalExpr(converted);
        if (val == nullptr || val->Count() != 1) {
            return nullptr;
        }
        return lConvertConstExpr(val, elementType, expr->pos);
    };

    if (collection != nullptr) {
        int elementCount = collection->GetElementCount();
        if (nInits > elementCount) {
            std::string name = "aggregate";
            if (CastType<StructType>(baseType) != nullptr) {
                name = "struct";
            } else if (CastType<ArrayType>(baseType) != nullptr) {
                name = "array";
            } else if (CastType<VectorType>(baseType) != nullptr) {
                name = "vector";
            }
            Error(exprList->pos,
                  "Initializer for %s type \"%s\" requires no more than %d values; %d provided.", name.c_str(),
                  expectedType->GetString().c_str(), elementCount, nInits);
            SetError();
            return nullptr;
        }

        std::vector<ConstExpr *> elements;
        elements.reserve(elementCount);
        for (int i = 0; i < elementCount; ++i) {
            const Type *elementType = collection->GetElementType(i);
            if (elementType == nullptr) {
                SetError();
                return nullptr;
            }
            ConstExpr *val = nullptr;
            if (i < nInits) {
                Expr *expr = exprList->exprs[i];
                if (auto *elist = llvm::dyn_cast_or_null<ExprList>(expr)) {
                    if (CastType<CollectionType>(elementType->GetAsNonConstType()) == nullptr) {
                        SetError();
                        return nullptr;
                    }
                    val = EvalExprList(elist, elementType);
                } else {
                    val = EvalExpr(expr);
                    if (val != nullptr) {
                        val = lConvertConstExpr(val, elementType, expr->pos);
                    }
                }
            } else {
                val = lMakeZeroConstExpr(elementType, exprList->pos);
            }
            if (val == nullptr) {
                SetError();
                return nullptr;
            }
            elements.push_back(val);
        }
        return new ConstExpr(expectedType, elements, exprList->pos);
    }

    if (baseType->IsVaryingType() && (atomic != nullptr || enumType != nullptr)) {
        const int width = g->target->getVectorWidth();
        if (nInits > width) {
            Error(exprList->pos,
                  "Initializer for varying type \"%s\" requires no more than %d values; %d provided.",
                  expectedType->GetString().c_str(), width, nInits);
            SetError();
            return nullptr;
        }
        if (nInits < width && nInits != 1) {
            Error(exprList->pos,
                  "Initializer for varying type \"%s\" requires %d values; %d provided.",
                  expectedType->GetString().c_str(), width, nInits);
            SetError();
            return nullptr;
        }

        const Type *elementType = expectedType->GetAsUniformType();
        AtomicType::BasicType basicType = atomic ? atomic->basicType : AtomicType::TYPE_UINT32;
        switch (basicType) {
        case AtomicType::TYPE_BOOL: {
            bool vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_INT8: {
            int8_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_UINT8: {
            uint8_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_INT16: {
            int16_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_UINT16: {
            uint16_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_INT32: {
            int32_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_UINT32: {
            uint32_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_INT64: {
            int64_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_UINT64: {
            uint64_t vals[ISPC_MAX_NVEC];
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                ce->GetValues(&vals[i]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals[i] = vals[0];
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        case AtomicType::TYPE_FLOAT16:
        case AtomicType::TYPE_FLOAT:
        case AtomicType::TYPE_DOUBLE: {
            std::vector<llvm::APFloat> vals;
            vals.reserve(width);
            llvm::Type *llvmElementType = elementType->LLVMType(g->ctx);
            for (int i = 0; i < nInits; ++i) {
                ConstExpr *ce = evalElement(exprList->exprs[i], elementType);
                if (ce == nullptr) {
                    SetError();
                    return nullptr;
                }
                std::vector<llvm::APFloat> tmp;
                ce->GetValues(tmp, llvmElementType);
                vals.push_back(tmp[0]);
            }
            if (nInits == 1) {
                for (int i = 1; i < width; ++i) {
                    vals.push_back(vals[0]);
                }
            }
            return new ConstExpr(expectedType, vals, exprList->pos);
        }
        default:
            break;
        }
    }

    if (atomic == nullptr && enumType == nullptr) {
        return nullptr;
    }

    if (nInits == 1) {
        ConstExpr *val = evalElement(exprList->exprs[0], expectedType);
        if (val == nullptr) {
            SetError();
            return nullptr;
        }
        return lConvertConstExpr(val, expectedType, exprList->exprs[0]->pos);
    }

    if (nInits > 1) {
        return nullptr;
    }
    SetError();
    return nullptr;
}

EvalResult ConstexprEvaluator::EvalStmtList(const StmtList *stmtList) {
    if (stmtList == nullptr) {
        return {EvalResult::Kind::Ok, nullptr};
    }
    PushScope();
    for (auto *stmt : stmtList->stmts) {
        EvalResult res = EvalStmt(stmt);
        if (res.kind != EvalResult::Kind::Ok) {
            PopScope();
            return res;
        }
    }
    PopScope();
    return {EvalResult::Kind::Ok, nullptr};
}

EvalResult ConstexprEvaluator::EvalStmt(const Stmt *stmt) {
    if (stmt == nullptr) {
        return {EvalResult::Kind::Ok, nullptr};
    }
    if (hasError) {
        return {EvalResult::Kind::Error, nullptr};
    }
    if (++steps > kMaxConstexprSteps) {
        SetError();
        return {EvalResult::Kind::Error, nullptr};
    }

    if (auto *sl = llvm::dyn_cast<StmtList>(stmt)) {
        return EvalStmtList(sl);
    }
    if (auto *cs = llvm::dyn_cast<CaseStmt>(stmt)) {
        return EvalStmt(cs->stmts);
    }
    if (auto *ds = llvm::dyn_cast<DefaultStmt>(stmt)) {
        return EvalStmt(ds->stmts);
    }
    if (auto *ds = llvm::dyn_cast<DeclStmt>(stmt)) {
        for (const auto &var : ds->vars) {
            if (var.sym == nullptr || var.init == nullptr) {
                SetError();
                return {EvalResult::Kind::Error, nullptr};
            }
            ConstExpr *val = nullptr;
            if (auto *elist = llvm::dyn_cast<ExprList>(var.init)) {
                val = EvalExprList(elist, var.sym->type);
            } else {
                val = EvalExpr(var.init);
            }
            if (val == nullptr) {
                SetError();
                return {EvalResult::Kind::Error, nullptr};
            }
            val = lConvertConstExpr(val, var.sym->type, var.sym->pos);
            if (val == nullptr) {
                SetError();
                return {EvalResult::Kind::Error, nullptr};
            }
            DeclareValue(var.sym, val);
        }
        return {EvalResult::Kind::Ok, nullptr};
    }
    if (auto *es = llvm::dyn_cast<ExprStmt>(stmt)) {
        if (es->expr != nullptr) {
            ConstExpr *val = EvalExpr(es->expr);
            if (val == nullptr) {
                SetError();
                return {EvalResult::Kind::Error, nullptr};
            }
        }
        return {EvalResult::Kind::Ok, nullptr};
    }
    if (auto *rs = llvm::dyn_cast<ReturnStmt>(stmt)) {
        if (rs->expr == nullptr) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }
        ConstExpr *val = EvalExpr(rs->expr);
        if (val == nullptr) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }
        return {EvalResult::Kind::Return, val};
    }
    if (auto *us = llvm::dyn_cast<UnmaskedStmt>(stmt)) {
        return EvalStmt(us->stmts);
    }
    if (auto *ifs = llvm::dyn_cast<IfStmt>(stmt)) {
        ConstExpr *cond = EvalExpr(ifs->test);
        if (cond == nullptr) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }
        cond = lConvertConstExpr(cond, AtomicType::UniformBool, ifs->test->pos);
        if (cond == nullptr) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }
        bool cv[ISPC_MAX_NVEC];
        int count = cond->GetValues(cv);
        if (count != 1) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }
        if (cv[0]) {
            return EvalStmt(ifs->trueStmts);
        }
        return EvalStmt(ifs->falseStmts);
    }
    if (auto *ss = llvm::dyn_cast<SwitchStmt>(stmt)) {
        ConstExpr *cond = EvalExpr(ss->expr);
        if (cond == nullptr) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }
        cond = lConvertConstExpr(cond, AtomicType::UniformInt64, ss->expr->pos);
        if (cond == nullptr) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }
        int64_t iv[ISPC_MAX_NVEC];
        int count = cond->GetValues(iv);
        if (count != 1) {
            SetError();
            return {EvalResult::Kind::Error, nullptr};
        }

        std::vector<const Stmt *> stmts;
        if (auto *sl = llvm::dyn_cast<StmtList>(ss->stmts)) {
            stmts.assign(sl->stmts.begin(), sl->stmts.end());
        } else if (ss->stmts != nullptr) {
            stmts.push_back(ss->stmts);
        }

        int startIndex = -1;
        int defaultIndex = -1;
        for (int i = 0; i < (int)stmts.size(); ++i) {
            if (auto *cs = llvm::dyn_cast<CaseStmt>(stmts[i])) {
                if (cs->value == (int)iv[0]) {
                    startIndex = i;
                    break;
                }
            } else if (defaultIndex < 0 && llvm::dyn_cast<DefaultStmt>(stmts[i])) {
                defaultIndex = i;
            }
        }
        if (startIndex < 0) {
            startIndex = defaultIndex;
        }
        if (startIndex < 0) {
            return {EvalResult::Kind::Ok, nullptr};
        }

        bool scoped = llvm::isa<StmtList>(ss->stmts);
        if (scoped) {
            PushScope();
        }
        for (int i = startIndex; i < (int)stmts.size(); ++i) {
            const Stmt *cur = stmts[i];
            EvalResult res;
            if (auto *cs = llvm::dyn_cast<CaseStmt>(cur)) {
                res = EvalStmt(cs->stmts);
            } else if (auto *ds = llvm::dyn_cast<DefaultStmt>(cur)) {
                res = EvalStmt(ds->stmts);
            } else {
                res = EvalStmt(cur);
            }
            if (res.kind == EvalResult::Kind::Return || res.kind == EvalResult::Kind::Error ||
                res.kind == EvalResult::Kind::Continue) {
                if (scoped) {
                    PopScope();
                }
                return res;
            }
            if (res.kind == EvalResult::Kind::Break) {
                if (scoped) {
                    PopScope();
                }
                return {EvalResult::Kind::Ok, nullptr};
            }
        }
        if (scoped) {
            PopScope();
        }
        return {EvalResult::Kind::Ok, nullptr};
    }
    if (auto *fs = llvm::dyn_cast<ForStmt>(stmt)) {
        PushScope();
        if (fs->init != nullptr) {
            EvalResult initRes = EvalStmt(fs->init);
            if (initRes.kind != EvalResult::Kind::Ok) {
                PopScope();
                return initRes;
            }
        }
        while (true) {
            if (fs->test != nullptr) {
                ConstExpr *cond = EvalExpr(fs->test);
                if (cond == nullptr) {
                    SetError();
                    PopScope();
                    return {EvalResult::Kind::Error, nullptr};
                }
                cond = lConvertConstExpr(cond, AtomicType::UniformBool, fs->test->pos);
                if (cond == nullptr) {
                    SetError();
                    PopScope();
                    return {EvalResult::Kind::Error, nullptr};
                }
                bool cv[ISPC_MAX_NVEC];
                int count = cond->GetValues(cv);
                if (count != 1) {
                    SetError();
                    PopScope();
                    return {EvalResult::Kind::Error, nullptr};
                }
                if (!cv[0]) {
                    break;
                }
            }
            EvalResult bodyRes = EvalStmt(fs->stmts);
            if (bodyRes.kind == EvalResult::Kind::Return) {
                PopScope();
                return bodyRes;
            }
            if (bodyRes.kind == EvalResult::Kind::Break) {
                break;
            }
            if (bodyRes.kind == EvalResult::Kind::Error) {
                PopScope();
                return bodyRes;
            }
            if (fs->step != nullptr) {
                EvalResult stepRes = EvalStmt(fs->step);
                if (stepRes.kind == EvalResult::Kind::Return) {
                    PopScope();
                    return stepRes;
                }
                if (stepRes.kind == EvalResult::Kind::Error) {
                    PopScope();
                    return stepRes;
                }
            }
        }
        PopScope();
        return {EvalResult::Kind::Ok, nullptr};
    }
    if (auto *ds = llvm::dyn_cast<DoStmt>(stmt)) {
        PushScope();
        while (true) {
            EvalResult bodyRes = EvalStmt(ds->bodyStmts);
            if (bodyRes.kind == EvalResult::Kind::Return) {
                PopScope();
                return bodyRes;
            }
            if (bodyRes.kind == EvalResult::Kind::Break) {
                break;
            }
            if (bodyRes.kind == EvalResult::Kind::Error) {
                PopScope();
                return bodyRes;
            }
            ConstExpr *cond = EvalExpr(ds->testExpr);
            if (cond == nullptr) {
                SetError();
                PopScope();
                return {EvalResult::Kind::Error, nullptr};
            }
            cond = lConvertConstExpr(cond, AtomicType::UniformBool, ds->testExpr->pos);
            if (cond == nullptr) {
                SetError();
                PopScope();
                return {EvalResult::Kind::Error, nullptr};
            }
            bool cv[ISPC_MAX_NVEC];
            int count = cond->GetValues(cv);
            if (count != 1) {
                SetError();
                PopScope();
                return {EvalResult::Kind::Error, nullptr};
            }
            if (!cv[0]) {
                break;
            }
        }
        PopScope();
        return {EvalResult::Kind::Ok, nullptr};
    }
    if (llvm::isa<BreakStmt>(stmt)) {
        return {EvalResult::Kind::Break, nullptr};
    }
    if (llvm::isa<ContinueStmt>(stmt)) {
        return {EvalResult::Kind::Continue, nullptr};
    }

    SetError();
    return {EvalResult::Kind::Error, nullptr};
}

ConstExpr *ConstexprEvaluator::EvalFunction(const Function *func, const std::vector<ConstExpr *> &args) {
    if (func == nullptr || func->GetCode() == nullptr) {
        if (allowDeferred) {
            SetDeferred();
            return nullptr;
        }
        SetError();
        return nullptr;
    }
    const FunctionType *ft = func->GetType();
    if (ft == nullptr || !ft->IsConstexpr()) {
        SetError();
        return nullptr;
    }
    if ((int)args.size() != ft->GetNumParameters()) {
        SetError();
        return nullptr;
    }

    PushScope();
    const std::vector<Symbol *> &params = func->GetParameterSymbols();
    for (int i = 0; i < ft->GetNumParameters(); ++i) {
        if (i < (int)params.size() && params[i] != nullptr) {
            ConstExpr *val = lConvertConstExpr(args[i], ft->GetParameterType(i), func->GetCode()->pos);
            if (val == nullptr) {
                SetError();
                PopScope();
                return nullptr;
            }
            DeclareValue(params[i], val);
        }
    }
    EvalResult res = EvalStmt(func->GetCode());
    PopScope();
    if (res.kind != EvalResult::Kind::Return || res.value == nullptr) {
        SetError();
        return nullptr;
    }
    ConstExpr *converted = lConvertConstExpr(res.value, ft->GetReturnType(), func->GetCode()->pos);
    if (converted == nullptr) {
        SetError();
        return nullptr;
    }
    return converted;
}

} // namespace

bool ispc::IsConstexprTypeAllowed(const Type *type) {
    if (type == nullptr) {
        return false;
    }
    const Type *t = type->GetAsNonConstType();
    if (CastType<AtomicType>(t) != nullptr || CastType<EnumType>(t) != nullptr || CastType<PointerType>(t) != nullptr) {
        return true;
    }
    if (const VectorType *vt = CastType<VectorType>(t)) {
        const Type *elt = vt->GetElementType()->GetAsNonConstType();
        return CastType<AtomicType>(elt) != nullptr || CastType<EnumType>(elt) != nullptr;
    }
    if (const ArrayType *at = CastType<ArrayType>(t)) {
        if (at->IsUnsized()) {
            return false;
        }
        return IsConstexprTypeAllowed(at->GetElementType());
    }
    if (const StructType *st = CastType<StructType>(t)) {
        for (int i = 0; i < st->GetElementCount(); ++i) {
            if (!IsConstexprTypeAllowed(st->GetElementType(i))) {
                return false;
            }
        }
        return true;
    }
    return false;
}

ConstexprEvalResult ispc::ConstexprEvaluateDetailed(Expr *expr, const Type *expectedType, bool allowDeferred) {
    ConstexprEvalResult result{nullptr, false, false};
    if (expr == nullptr) {
        return result;
    }
    ConstexprEvaluator eval(0, allowDeferred);
    ConstExpr *value = nullptr;
    if (auto *elist = llvm::dyn_cast<ExprList>(expr)) {
        value = eval.EvalExprList(elist, expectedType);
    } else {
        value = eval.EvalExpr(expr);
    }
    if (value == nullptr || eval.Failed()) {
        result.deferred = eval.Deferred();
        return result;
    }
    if (expectedType != nullptr) {
        value = lConvertConstExpr(value, expectedType, expr->pos);
    }
    result.value = value;
    result.deferred = eval.Deferred();
    result.targetDependent = eval.IsTargetDependent();
    return result;
}

ConstExpr *ispc::ConstexprEvaluate(Expr *expr, const Type *expectedType) {
    ConstexprEvalResult result = ConstexprEvaluateDetailed(expr, expectedType, false);
    return result.value;
}

namespace {

class ConstexprValidator {
  public:
    explicit ConstexprValidator(const Function *f) : func(f), ok(true) {
        if (func) {
            funcName = func->GetName();
        }
    }

    bool Validate();

  private:
    const Function *func;
    std::string funcName;
    bool ok;
    std::unordered_set<const Symbol *> locals;

    void GatherLocals(const Stmt *stmt);
    void VisitStmt(const Stmt *stmt);
    void VisitExpr(const Expr *expr);

    bool IsLocalOrParam(const Symbol *sym) const {
        if (sym == nullptr) {
            return false;
        }
        if (sym->GetSymbolKind() == Symbol::SymbolKind::FunctionParm) {
            return true;
        }
        return locals.find(sym) != locals.end();
    }

    void DisallowedStmt(const Stmt *stmt, const char *what) {
        if (!stmt || !func) {
            return;
        }
        Error(stmt->pos, "constexpr function \"%s\" contains disallowed statement \"%s\".", funcName.c_str(), what);
        ok = false;
    }

    void DisallowedExpr(const Expr *expr, const char *what) {
        if (!expr || !func) {
            return;
        }
        Error(expr->pos, "constexpr function \"%s\" contains disallowed expression \"%s\".", funcName.c_str(), what);
        ok = false;
    }
};

void ConstexprValidator::GatherLocals(const Stmt *stmt) {
    if (stmt == nullptr) {
        return;
    }
    if (auto *ds = llvm::dyn_cast<DeclStmt>(stmt)) {
        for (const auto &var : ds->vars) {
            if (var.sym) {
                locals.insert(var.sym);
            }
        }
    }
    if (auto *sl = llvm::dyn_cast<StmtList>(stmt)) {
        for (auto *s : sl->stmts) {
            GatherLocals(s);
        }
    } else if (auto *ifs = llvm::dyn_cast<IfStmt>(stmt)) {
        GatherLocals(ifs->trueStmts);
        GatherLocals(ifs->falseStmts);
    } else if (auto *fs = llvm::dyn_cast<ForStmt>(stmt)) {
        GatherLocals(fs->init);
        GatherLocals(fs->step);
        GatherLocals(fs->stmts);
    } else if (auto *ds = llvm::dyn_cast<DoStmt>(stmt)) {
        GatherLocals(ds->bodyStmts);
    } else if (auto *ss = llvm::dyn_cast<SwitchStmt>(stmt)) {
        GatherLocals(ss->stmts);
    } else if (auto *cs = llvm::dyn_cast<CaseStmt>(stmt)) {
        GatherLocals(cs->stmts);
    } else if (auto *ds = llvm::dyn_cast<DefaultStmt>(stmt)) {
        GatherLocals(ds->stmts);
    } else if (auto *us = llvm::dyn_cast<UnmaskedStmt>(stmt)) {
        GatherLocals(us->stmts);
    } else if (auto *ls = llvm::dyn_cast<LabeledStmt>(stmt)) {
        GatherLocals(ls->stmt);
    }
}

void ConstexprValidator::VisitExpr(const Expr *expr) {
    if (expr == nullptr) {
        return;
    }
    if (llvm::dyn_cast<NewExpr>(expr)) {
        DisallowedExpr(expr, "new");
        return;
    }
    if (llvm::dyn_cast<SyncExpr>(expr)) {
        DisallowedExpr(expr, "sync");
        return;
    }
    if (llvm::dyn_cast<AllocaExpr>(expr)) {
        DisallowedExpr(expr, "alloca");
        return;
    }
    if (auto *fce = llvm::dyn_cast<FunctionCallExpr>(expr)) {
        const FunctionSymbolExpr *fse = llvm::dyn_cast<FunctionSymbolExpr>(fce->func);
        if (fse == nullptr || fse->GetBaseSymbol() == nullptr) {
            if (fse == nullptr) {
                DisallowedExpr(expr, "function pointer call");
            }
            return;
        }
        Symbol *callee = fse->GetBaseSymbol();
        if (!callee->isConstexpr) {
            Error(expr->pos, "constexpr function \"%s\" calls non-constexpr function \"%s\".", funcName.c_str(),
                  callee->name.c_str());
            ok = false;
            return;
        }
    }
    if (auto *ae = llvm::dyn_cast<AssignExpr>(expr)) {
        const SymbolExpr *lhs = llvm::dyn_cast<SymbolExpr>(ae->lvalue);
        Symbol *sym = lhs ? lhs->GetBaseSymbol() : nullptr;
        if (lhs == nullptr || sym == nullptr || !IsLocalOrParam(sym) || sym->storageClass.IsStatic()) {
            DisallowedExpr(expr, "assignment");
            return;
        }
    }
    if (auto *ue = llvm::dyn_cast<UnaryExpr>(expr)) {
        if (ue->op == UnaryExpr::PreInc || ue->op == UnaryExpr::PreDec ||
            ue->op == UnaryExpr::PostInc || ue->op == UnaryExpr::PostDec) {
            const SymbolExpr *lhs = llvm::dyn_cast<SymbolExpr>(ue->expr);
            Symbol *sym = lhs ? lhs->GetBaseSymbol() : nullptr;
            if (lhs == nullptr || sym == nullptr || !IsLocalOrParam(sym) || sym->storageClass.IsStatic()) {
                DisallowedExpr(expr, "increment/decrement");
                return;
            }
        }
    }

    if (auto *ue = llvm::dyn_cast<UnaryExpr>(expr)) {
        VisitExpr(ue->expr);
    } else if (auto *be = llvm::dyn_cast<BinaryExpr>(expr)) {
        VisitExpr(be->arg0);
        VisitExpr(be->arg1);
    } else if (auto *ae = llvm::dyn_cast<AssignExpr>(expr)) {
        VisitExpr(ae->lvalue);
        VisitExpr(ae->rvalue);
    } else if (auto *el = llvm::dyn_cast<ExprList>(expr)) {
        for (auto *item : el->exprs) {
            VisitExpr(item);
        }
    } else if (auto *se = llvm::dyn_cast<SelectExpr>(expr)) {
        VisitExpr(se->test);
        VisitExpr(se->expr1);
        VisitExpr(se->expr2);
    } else if (auto *ie = llvm::dyn_cast<IndexExpr>(expr)) {
        const Type *baseType = ie->baseExpr ? ie->baseExpr->GetType() : nullptr;
        if (baseType != nullptr && !baseType->IsDependent() && CastType<PointerType>(baseType) != nullptr) {
            DisallowedExpr(expr, "pointer dereference");
            return;
        }
        VisitExpr(ie->baseExpr);
        VisitExpr(ie->index);
    } else if (auto *me = llvm::dyn_cast<MemberExpr>(expr)) {
        const Type *baseType = me->expr ? me->expr->GetType() : nullptr;
        if (baseType != nullptr && !baseType->IsDependent() && CastType<PointerType>(baseType) != nullptr) {
            DisallowedExpr(expr, "pointer dereference");
            return;
        }
        VisitExpr(me->expr);
    } else if (auto *tce = llvm::dyn_cast<TypeCastExpr>(expr)) {
        VisitExpr(tce->expr);
    } else if (auto *re = llvm::dyn_cast<ReferenceExpr>(expr)) {
        VisitExpr(re->expr);
    } else if (auto *pd = llvm::dyn_cast<PtrDerefExpr>(expr)) {
        DisallowedExpr(expr, "pointer dereference");
        VisitExpr(pd->expr);
    } else if (auto *rd = llvm::dyn_cast<RefDerefExpr>(expr)) {
        VisitExpr(rd->expr);
    } else if (auto *ao = llvm::dyn_cast<AddressOfExpr>(expr)) {
        Symbol *sym = ao->GetBaseSymbol();
        if (sym == nullptr || (sym->parentFunction != nullptr && sym->GetSymbolKind() != Symbol::SymbolKind::Function)) {
            DisallowedExpr(expr, "address-of");
            return;
        }
        VisitExpr(ao->expr);
    } else if (auto *fce = llvm::dyn_cast<FunctionCallExpr>(expr)) {
        VisitExpr(fce->func);
        if (fce->args) {
            for (auto *arg : fce->args->exprs) {
                VisitExpr(arg);
            }
        }
    }
}

void ConstexprValidator::VisitStmt(const Stmt *stmt) {
    if (stmt == nullptr) {
        return;
    }
    auto checkUniform = [this](const Expr *expr, const char *context) {
        if (expr == nullptr) {
            return;
        }
        const Type *t = expr->GetType();
        if (t == nullptr || t->IsDependent()) {
            return;
        }
        if (t->IsVaryingType()) {
            Error(expr->pos, "constexpr function \"%s\" requires a uniform condition in \"%s\".", funcName.c_str(),
                  context);
            ok = false;
        }
    };
    if (llvm::dyn_cast<PrintStmt>(stmt)) {
        DisallowedStmt(stmt, "print");
        return;
    }
    if (llvm::dyn_cast<AssertStmt>(stmt)) {
        DisallowedStmt(stmt, "assert");
        return;
    }
    if (llvm::dyn_cast<ForeachStmt>(stmt) || llvm::dyn_cast<ForeachActiveStmt>(stmt) ||
        llvm::dyn_cast<ForeachUniqueStmt>(stmt)) {
        DisallowedStmt(stmt, "foreach");
        return;
    }
    if (llvm::dyn_cast<GotoStmt>(stmt) || llvm::dyn_cast<LabeledStmt>(stmt)) {
        DisallowedStmt(stmt, "goto");
        return;
    }
    if (llvm::dyn_cast<DeleteStmt>(stmt)) {
        DisallowedStmt(stmt, "delete");
        return;
    }
    if (auto *rs = llvm::dyn_cast<ReturnStmt>(stmt)) {
        VisitExpr(rs->expr);
        return;
    }
    if (auto *es = llvm::dyn_cast<ExprStmt>(stmt)) {
        VisitExpr(es->expr);
        return;
    }
    if (auto *ds = llvm::dyn_cast<DeclStmt>(stmt)) {
        for (const auto &var : ds->vars) {
            if (var.init == nullptr) {
                const char *name = var.sym ? var.sym->name.c_str() : "<unnamed>";
                Error(stmt->pos, "constexpr function \"%s\" requires initializer for local variable \"%s\".",
                      funcName.c_str(), name);
                ok = false;
            }
            VisitExpr(var.init);
        }
        return;
    }
    if (auto *sl = llvm::dyn_cast<StmtList>(stmt)) {
        for (auto *s : sl->stmts) {
            VisitStmt(s);
        }
        return;
    }
    if (auto *ifs = llvm::dyn_cast<IfStmt>(stmt)) {
        VisitExpr(ifs->test);
        checkUniform(ifs->test, "if");
        VisitStmt(ifs->trueStmts);
        VisitStmt(ifs->falseStmts);
        return;
    }
    if (auto *fs = llvm::dyn_cast<ForStmt>(stmt)) {
        VisitStmt(fs->init);
        VisitExpr(fs->test);
        checkUniform(fs->test, "loop");
        VisitStmt(fs->step);
        VisitStmt(fs->stmts);
        return;
    }
    if (auto *us = llvm::dyn_cast<UnmaskedStmt>(stmt)) {
        VisitStmt(us->stmts);
        return;
    }
    if (auto *ds = llvm::dyn_cast<DoStmt>(stmt)) {
        VisitExpr(ds->testExpr);
        checkUniform(ds->testExpr, "loop");
        VisitStmt(ds->bodyStmts);
        return;
    }
    if (auto *ss = llvm::dyn_cast<SwitchStmt>(stmt)) {
        VisitExpr(ss->expr);
        checkUniform(ss->expr, "switch");
        VisitStmt(ss->stmts);
        return;
    }
    if (auto *cs = llvm::dyn_cast<CaseStmt>(stmt)) {
        VisitStmt(cs->stmts);
        return;
    }
    if (auto *ds = llvm::dyn_cast<DefaultStmt>(stmt)) {
        VisitStmt(ds->stmts);
        return;
    }
}

bool ConstexprValidator::Validate() {
    if (func == nullptr) {
        return false;
    }
    const FunctionType *ft = func->GetType();
    if (ft == nullptr) {
        return false;
    }
    if (ft->GetReturnType()->IsVoidType()) {
        Error(ft->GetSourcePos(), "constexpr function \"%s\" must return a value.", funcName.c_str());
        ok = false;
    }
    const Stmt *body = func->GetCode();
    GatherLocals(body);
    VisitStmt(body);
    return ok;
}

} // namespace

bool ispc::ValidateConstexprFunction(const Function *func) {
    ConstexprValidator validator(func);
    return validator.Validate();
}
