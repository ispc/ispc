// Copyright 2026, Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

// A failed expression can be dropped before emitting a reference return.
// RUN: not %{ispc} %s -DSIZEOF --target=host --nostdlib --nowrap --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=SIZEOF --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"
// RUN: not %{ispc} %s -DDIVISION --target=host --nostdlib --nowrap --opt=fast-math:legacy --emit-llvm-text -o %t.ll 2>&1 | FileCheck %s --check-prefix=DIVISION --implicit-check-not="FATAL ERROR" --implicit-check-not="Assertion failed"

#ifdef SIZEOF
// SIZEOF: Error: "uniform" qualifier is illegal with "void" type.
uniform int &f(uniform int &r) { return sizeof(uniform void); }
#else
void rcp(uniform float x);

// DIVISION: Error: First operand to binary operator "*" is of invalid type "void".
uniform int64 &f(uniform float a, uniform float b, uniform float *uniform p, uniform int64 n) {
    return n / (uniform int64)&(a / b, p);
}
#endif
