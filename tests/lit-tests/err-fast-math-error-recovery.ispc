// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// A failed reciprocal rewrite can leave a non-null expression with a null type.
// Division and both branches of uniform/varying selects must recover normally.
// RUN: not %{ispc} %s -DDIVISION --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_FIRST_VARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_SECOND_VARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_FIRST_UNIFORM --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_SECOND_UNIFORM --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"

void rcp(uniform float x);

// CHECK: Error: First operand to binary operator "*" is of invalid type "void".
#if defined(SELECT_FIRST_UNIFORM) || defined(SELECT_SECOND_UNIFORM)
void test_recovery(uniform float a, uniform float b, uniform float * uniform p,
                   uniform int64 n, uniform bool condition) {
#else
void test_recovery(uniform float a, uniform float b, uniform float * uniform p,
                   uniform int64 n, varying bool condition) {
#endif
#ifdef DIVISION
    n / (uniform int64)&(a / b, p);
#elif defined(SELECT_FIRST_UNIFORM) || defined(SELECT_FIRST_VARYING)
    condition ? (uniform int64)&(a / b, p) : n;
#else
    condition ? n : (uniform int64)&(a / b, p);
#endif
}
