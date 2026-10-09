// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// A malformed function typedef must not crash signed/unsigned diagnostics.
// RUN: not %{ispc} %s -DUNSIGNED --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSIGNED --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSIZEOF_UNSIGNED --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DSIZEOF_SIGNED --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DCAST_SIGNED --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DCAST_UNSIGNED --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"

// CHECK: Error: Parameter with type "void" illegal in function parameter list.
typedef void F(void);

#ifdef UNSIGNED
unsigned F x;
#endif
#ifdef SIGNED
signed F x;
#endif
#ifdef SIZEOF_UNSIGNED
uniform int f() { return sizeof(unsigned F); }
#endif
#ifdef SIZEOF_SIGNED
uniform int f() { return sizeof(signed F); }
#endif
#ifdef CAST_SIGNED
void f() { (signed F)0; }
#endif
#ifdef CAST_UNSIGNED
void f() { (unsigned F)0; }
#endif
