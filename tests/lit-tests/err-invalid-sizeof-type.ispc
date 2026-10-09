// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// Invalid sizeof types must be discarded before select-expression emission.
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_TRUE_UNIFORM --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_FALSE_UNIFORM --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_TRUE_VARYING --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSELECT_FALSE_VARYING --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DTEMPLATE --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DINCOMPLETE_TRUE --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=INCOMPLETE --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DINCOMPLETE_FALSE --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=INCOMPLETE --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DINCOMPLETE_REFERENCE --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=INCOMPLETE --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"

// CHECK: Error: "uniform" qualifier is illegal with "void" type.
// INCOMPLETE: Error: Can't compute the size of declared but not defined struct type
#if defined(INCOMPLETE_REFERENCE)
struct S;
uniform int test_sizeof(uniform S &r) { return sizeof(r); }
#elif defined(INCOMPLETE_TRUE) || defined(INCOMPLETE_FALSE)
struct S;
uniform int test_sizeof(uniform bool condition, uniform S * uniform p) {
#ifdef INCOMPLETE_TRUE
    return condition ? sizeof(*p) : 1;
#else
    return condition ? 1 : sizeof(*p);
#endif
}
#else
#ifdef TEMPLATE
template <typename T> uniform int test_sizeof(T value) {
#elif defined(SELECT_TRUE_UNIFORM) || defined(SELECT_FALSE_UNIFORM)
uniform int test_sizeof(uniform bool condition) {
#elif defined(SELECT_TRUE_VARYING) || defined(SELECT_FALSE_VARYING)
int test_sizeof(bool condition) {
#else
uniform int test_sizeof() {
#endif
#if defined(SELECT_TRUE_UNIFORM) || defined(SELECT_TRUE_VARYING)
    return condition ? sizeof(uniform void) : 1;
#elif defined(SELECT_FALSE_UNIFORM) || defined(SELECT_FALSE_VARYING)
    return condition ? 1 : sizeof(uniform void);
#else
    return sizeof(uniform void);
#endif
}
#ifdef TEMPLATE
uniform int instantiate() { return test_sizeof<int>(0); }
#endif
#endif
