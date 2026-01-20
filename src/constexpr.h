/*
  Copyright (c) 2026, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

/** @file constexpr.h
    @brief Helpers for constexpr validation and evaluation.
*/

#pragma once

namespace ispc {

class ConstExpr;
class Expr;
class Function;
class Type;

struct ConstexprEvalResult {
    ConstExpr *value;
    bool deferred;
    bool targetDependent;
};

/** Returns true if the type is allowed for constexpr variables in v1. */
bool IsConstexprTypeAllowed(const Type *type);

/** Try to evaluate an expression to a ConstExpr using constexpr rules.
    Returns nullptr if the expression can't be constant-evaluated. */
ConstExpr *ConstexprEvaluate(Expr *expr, const Type *expectedType);

/** Evaluate an expression to a ConstExpr using constexpr rules, capturing
    deferral and target-dependent information. */
ConstexprEvalResult ConstexprEvaluateDetailed(Expr *expr, const Type *expectedType, bool allowDeferred);

/** Validate constexpr-suitable constructs in a function definition.
    Returns false if any errors were emitted. */
bool ValidateConstexprFunction(const Function *func);

} // namespace ispc
