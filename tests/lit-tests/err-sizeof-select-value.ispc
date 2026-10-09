// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// A failed optimization inside sizeof can leave a valid type but no value in
// either branch or the condition of a scalar/short-vector select.
// RUN: not %{ispc} %s -DFIRST --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DFIRST -DVARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DVARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DFIRST -DSHORT_VECTOR --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSHORT_VECTOR --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DFIRST -DSHORT_VECTOR -DVARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSHORT_VECTOR -DVARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DCONDITION --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DCONDITION -DVARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DCONDITION -DSHORT_VECTOR --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DCONDITION -DSHORT_VECTOR -DVARYING --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"

// CHECK: Error: First operand to binary operator "*" is of invalid type "void".
#ifdef SHORT_VECTOR
#define INT int<3>
#define BOOL bool<3>
#else
#define INT int
#define BOOL bool
#endif
#ifdef VARYING
#define VAR varying
#else
#define VAR uniform
#endif

void rcp(uniform float x);
VAR INT f(uniform float a, uniform float b, uniform float * uniform p, VAR BOOL c, VAR INT q) {
#ifdef CONDITION
    return (q + sizeof((uniform int64)&(a / b, p))) != 0 ? q : q + 1;
#elif defined(FIRST)
    return c ? sizeof((uniform int64)&(a / b, p)) : 1;
#else
    return c ? 1 : sizeof((uniform int64)&(a / b, p));
#endif
}
