//; RUN: %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8 2>&1 | FileCheck %s -check-prefix=CHECK_SINGLE
//; RUN: not %{ispc} %s -o %t.o --nowrap --target=avx2-i32x8,sse2-i32x4 2>&1 | FileCheck %s -check-prefix=CHECK_MULTI

// REQUIRES: X86_ENABLED

//; CHECK_SINGLE: Initializer for global variable "sz_var" is a constant for single-target compilation but not for multi-target compilation.
//; CHECK_MULTI: Initializer for global variable "sz_var" is not a constant for multi-target compilation
uniform int sz = sizeof(uniform int);

static uniform float a = 1.0;
static uniform float *uniform pa = &a;
static const uniform float *uniform parr[] = {NULL, &a, &a};

static const uniform float b[] = {0., 4., 3., 2., 1.};
static const uniform float *uniform pb[] = {NULL, b + 3, b};
static const uniform float *uniform pb1[] = {NULL, b, b};

varying float i = 80;
varying float p_foo[] = {1, 4, 3, 2, 4, 3, 3, 3};
varying float *uniform au = &p_foo[0];

uniform int sz_var = sizeof(varying int);

uniform int ii[2][3][4] = {{{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}}, {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}}};
uniform int *uniform aaa = &ii[1][2][3];

varying float usevals() {

    varying float *varying av = au + i + *pa * *parr[1] * *pb[2] * *pb1[1] + *aaa;
    return *av;
}
