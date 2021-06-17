//; RUN: %{ispc} %s --target=host --emit-llvm-text -O0 -o - | FileCheck %s

// Tests float16 constant parsing in *e*f16 form.
//; CHECK: define half @foo0___unh(half
//; CHECK: fadd half %{{[a-zA-Z_][a-zA-Z0-9_]*}}, 0xH011D
uniform float16 foo0(uniform float16 arg0) {
    arg0 = arg0 + 1.7e-5f16;
    return arg0;
}

// Tests float16 constant parsing in *.*F16 form.
//; CHECK: define void @foo1___REFunhun_3C_unh_3E_(half* noalias %arg0, half* noalias %arg1
//; CHECK: store half 0xH4252, half*
//; CHECK: fmul half %arg
void foo1(uniform float16 &arg0, uniform float16 *uniform arg1) {
    arg0 = 3.16F16;
    *arg1 = *arg1 * arg0;
}

// Tests simple control flow with float16
//; CHECK: define half @foo2___unhunh
//; CHECK: br
//; CHECK: ret half %
//; CHECK: ret half %
uniform float16 foo2(uniform float16 arg0, uniform float16 arg1) {
    if (arg0 > arg1)
        return arg0;
    else
        return arg1;
}

// Tests function call with float16 arguments.
//; CHECK: define half @foo3___unh(half
//; CHECK: call half @goo3___unh(half
noinline uniform float16 goo3(uniform float16 arg0) { return arg0 + 7.9; }

uniform float16 foo3(uniform float16 arg0) { return goo3(arg0); }

// Tests int type -> float16 type conversion.
//; CHECK: define half @foo4___uni(i32 %
//; CHECK: sitofp i32 %{{[a-zA-Z_][a-zA-Z0-9_]*}} to half
uniform float16 foo4(uniform int arg0) { return arg0; }

// Tests float16 -> wider float type conversion.
//; CHECK: define double @foo5___unh(half %arg0,
//; CHECK: fpext half %{{[a-zA-Z_][a-zA-Z0-9_]*}} to double
uniform double foo5(uniform float16 arg0) { return arg0; }

// Creates +0, -0, +inf and -inf.
//; CHECK: define void @foo6
//; CHECK: store half 0xH0000
//; CHECK: store half 0xH8000
//; CHECK: store half 0xH7C00
//; CHECK: store half 0xHFC00
void foo6() {
    uniform float16 pz = +0.f16;
    uniform float16 nz = -0.f16;
    uniform float16 pinf = 0x7c0016;
    uniform float16 ninf = -0xfc0016;
    return;
}

// Testing precedence for float16 with 'higher' precedence type.
//; CHECK: define i8 @foo7___unfunh
//; CHECK: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = fpext half %{{[a-zA-Z_][a-zA-Z0-9_]*}} to float
//; CHECK: fadd float
uniform int8 foo7(uniform float arg0, uniform float16 arg1) {
    uniform int8 arg = arg0 + arg1;
    return arg;
}

// Testing precedence for float16 with 'lower' precedence type.
//; CHECK: define i8 @foo8___unsunh(
//; CHECK: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = sitofp i16 %{{[a-zA-Z_][a-zA-Z0-9_]*}} to half
//; CHECK: fadd half
uniform int8 foo8(uniform int16 arg0, uniform float16 arg1) {
    uniform int8 arg = arg0 + arg1;
    return arg;
}
