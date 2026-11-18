// Copyright (c) 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// An ordinary user definition can still replace an implicitly included library overload.
// RUN: %{ispc} %s --target=host --nowrap --emit-llvm-text -o - | FileCheck %s
// RUN: not %{ispc} %s --target=host --nowrap --emit-llvm-text -DCONSTEXPR_CALL -o %t.ll 2>&1 | FileCheck %s --check-prefix=CONSTEXPR
// RUN: not %{ispc} %s --target=host --nowrap --emit-llvm-text -DCONFLICTING_DECLARATION -o %t.ll 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: not %{ispc} %s --target=host --nowrap --emit-llvm-text -DDUPLICATE_DEFINITION -o %t.ll 2>&1 | FileCheck %s --check-prefix=DUPLICATE
// RUN: not %{ispc} %s --target=host --nowrap --emit-llvm-text -DCHANGED_ABI -o %t.ll 2>&1 | FileCheck %s --check-prefix=ABI
// RUN: not %{ispc} %s --target=host --nowrap --emit-llvm-text -DLINE_DIRECTIVE -o %t.ll 2>&1 | FileCheck %s --check-prefix=LINE
// RUN: %{ispc} %s --target=host --nowrap --emit-llvm-text -DOVERRIDE_NUMERIC_HELPERS -o - | FileCheck %s --check-prefix=HELPERS

#ifdef OVERRIDE_NUMERIC_HELPERS
// Replacing a public overload leaves the other library implementations independent.
uniform uint32 intbits(uniform float x) { return 0x7f800000u; }
uniform double abs(uniform double x) { return doublebits(0x7ff0000000000000ull); }

static_assert(isfinite(1.0f) && !isinf(1.0f) && !isinf(1.0d));
static_assert(abs(-1.0f) == 1.0f && signbits(-1.0f) == 0x80000000u);

// HELPERS-LABEL: define {{.*}} @library_classification___
// HELPERS: ret i32 1
uniform int library_classification() {
    return isfinite(1.0f) && !isinf(1.0f) && !isinf(1.0d) && abs(-1.0f) == 1.0f && signbits(-1.0f) == 0x80000000u;
}

#else

// CHECK-LABEL: define {{.*}} @call_before___
// CHECK: ret i32 42
uniform int call_before(uniform float x) { return isfinite(x) ? 1 : 42; }

#ifdef CHANGED_ABI
// ABI: Error: Function "isfinite" redeclared with different "constexpr" qualifier
__attribute__((unmangled)) uniform bool isfinite(uniform float x);
#endif

uniform bool isfinite(uniform float x);

#ifdef CONFLICTING_DECLARATION
// CONFLICT: Error: Function "isfinite" redeclared with different "constexpr" qualifier
constexpr uniform bool isfinite(uniform float x);
#endif

uniform bool isfinite(uniform float x) { return false; }
varying bool isfinite(varying float x) { return false; }

// CHECK-LABEL: define {{.*}} @call_after___
// CHECK: ret i32 42
uniform int call_after(uniform float x) { return isfinite(x) ? 1 : 42; }

// A prototype alone still gets the runtime implementation from the library.
uniform bool isfinite(uniform double x);

// CHECK-LABEL: define {{.*}} @call_declared___
// CHECK: ret i32 1
uniform int call_declared() { return isfinite(1.0d) ? 1 : 42; }

#ifdef CONSTEXPR_CALL
// CONSTEXPR: Error: static_assert condition must be a compile-time constant.
static_assert(isfinite(1.0f));
#endif

#ifdef DUPLICATE_DEFINITION
// DUPLICATE: Error: Ignoring redefinition of function "isfinite".
uniform bool isfinite(uniform float x) { return true; }
#endif

#ifdef LINE_DIRECTIVE
#line 1 "/stdlib.isph"
constexpr uniform int user_function() { return 1; }
// LINE: Error: Function "user_function" redeclared with different "constexpr" qualifier
uniform int user_function() { return 2; }
#endif

#endif
