// RUN: %{ispc} %s --target=sse2-i32x4 --emit-llvm-text -O0 -o - | FileCheck %s -check-prefix=CHECK_FLOAT
// RUN: %{ispc} %s --target=sse4-i32x4 --emit-llvm-text -O0 -o - | FileCheck %s -check-prefix=CHECK_FLOAT
// RUN: %{ispc} %s --target=avx1-i32x4 --emit-llvm-text -O0 -o - | FileCheck %s -check-prefix=CHECK_FLOAT

// RUN: %{ispc} %s --target=avx2-i32x4 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF
// RUN: %{ispc} %s --target=avx512knl-x16 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF
// RUN: %{ispc} %s --target=avx512skx-x4 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF
// RUN: %{ispc} %s --target=neon-i32x4 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF

// REQUIRES: X86_ENABLED

// Tests float16 constant parsing in *e*f16 form.
//; CHECK_HALF: define half @foo0___unh(half
//; CHECK_HALF: fadd half %{{[a-zA-Z_][a-zA-Z0-9_]*}}, 0xH011D
//; CHECK_FLOAT: define half @foo0___unh(half
//; CHECK_FLOAT: fadd float %{{[a-zA-Z_\.][a-zA-Z0-9_\.]*}}
uniform float16 foo0(uniform float16 arg0) {
    arg0 = arg0 + 1.7e-5f16;
    return arg0;
}

// Tests float16 constant parsing in *.*F16 form.
//; CHECK_HALF: define void @foo1___REFunhun_3C_unh_3E_
//; CHECK_HALF: store half 0xH4252,
//; CHECK_HALF: fmul half %arg
void foo1(uniform float16 &arg0, uniform float16 *uniform arg1) {
    arg0 = 3.16F16;
    *arg1 = *arg1 * arg0;
}

// Tests simple control flow with float16
//; CHECK_HALF: define half @foo2___unhunh
//; CHECK_HALF: br
//; CHECK_HALF: ret half %
//; CHECK_HALF: ret half %
uniform float16 foo2(uniform float16 arg0, uniform float16 arg1) {
    if (arg0 > arg1)
        return arg0;
    else
        return arg1;
}

// Tests function call with float16 arguments.
//; CHECK_HALF: define half @foo3___unh(half
//; CHECK_HALF: call half @goo3___unh(half
noinline uniform float16 goo3(uniform float16 arg0) { return arg0 + 7.9; }

uniform float16 foo3(uniform float16 arg0) { return goo3(arg0); }

// Tests int type -> float16 type conversion.
//; CHECK_HALF: define half @foo4___uni(i32 %
//; CHECK_HALF: sitofp i32 %{{[a-zA-Z_][a-zA-Z0-9_]*}} to half
uniform float16 foo4(uniform int arg0) { return arg0; }

// Tests float16 -> wider float type conversion.
//; CHECK_HALF: define double @foo5___unh(half %arg0,
//; CHECK_HALF: fpext half %{{[a-zA-Z_][a-zA-Z0-9_]*}} to double
uniform double foo5(uniform float16 arg0) { return arg0; }

// Creates +0, -0, +inf and -inf.
//; CHECK_HALF: define void @foo6
//; CHECK_HALF: store half 0xH0000
//; CHECK_HALF: store half 0xH8000
//; CHECK_HALF: store half 0xH7C00
//; CHECK_HALF: store half 0xHFC00
void foo6() {
    uniform float16 pz = +0.f16;
    uniform float16 nz = -0.f16;
    uniform float16 pinf = 0x7c0016;
    uniform float16 ninf = -0xfc0016;
    return;
}

// Testing precedence for float16 with 'higher' precedence type.
//; CHECK_HALF: define i8 @foo7___unfunh
//; CHECK_HALF: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = fpext half %{{[a-zA-Z_][a-zA-Z0-9_]*}} to float
//; CHECK_HALF: fadd float
uniform int8 foo7(uniform float arg0, uniform float16 arg1) {
    uniform int8 arg = arg0 + arg1;
    return arg;
}

// Testing precedence for float16 with 'lower' precedence type.
//; CHECK_HALF: define i8 @foo8___unsunh(
//; CHECK_HALF: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = sitofp i16 %{{[a-zA-Z_][a-zA-Z0-9_]*}} to half
//; CHECK_HALF: fadd half
uniform int8 foo8(uniform int16 arg0, uniform float16 arg1) {
    uniform int8 arg = arg0 + arg1;
    return arg;
}
