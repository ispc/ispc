// RUN: %{ispc} %s --target=sse2-i32x4 --emit-llvm-text -O0 -o - | FileCheck %s -check-prefix=CHECK_FLOAT
// RUN: %{ispc} %s --target=sse4-i32x4 --emit-llvm-text -O0 -o - | FileCheck %s -check-prefix=CHECK_FLOAT
// RUN: %{ispc} %s --target=avx1-i32x4 --emit-llvm-text -O0 -o - | FileCheck %s -check-prefix=CHECK_FLOAT

// RUN: %{ispc} %s --target=avx2-i32x4 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF
// RUN: %{ispc} %s --target=avx512knl-x16 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF
// RUN: %{ispc} %s --target=avx512skx-x4 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF
// RUN: %{ispc} %s --target=neon-i32x4 -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_HALF

// REQUIRES: X86_ENABLED

// Tests float16 constant parsing in *e*f16 form.
//; CHECK_HALF: define <{{[0-9]*}} x half> @foo0___vyh(<{{[0-9]*}} x half>
//; CHECK_HALF: fadd <{{[0-9]*}} x half> %{{[a-zA-Z_][a-zA-Z0-9_]*}}, <half 0xH011D, half 0xH011D
//; CHECK_FLOAT: define <{{[0-9]*}} x half> @foo0___vyh(<{{[0-9]*}} x half>
//; CHECK_FLOAT: fadd <{{[0-9]*}} x float> %{{[a-zA-Z_\.][a-zA-Z0-9_\.]*}}, <float
varying float16 foo0(varying float16 arg0) {
    arg0 = arg0 + 1.7e-5f16;
    return arg0;
}

// Tests float16 constant parsing in *.*f16 form.
//; CHECK_HALF: define void @foo1___REFvyhvy_3C_vyh_3E_({{.*}} noalias %arg0, <{{[0-9]*}} x i64> %arg1
//; CHECK_HALF: fmul <{{[0-9]*}} x half>
void foo1(varying float16 &arg0, varying float16 *varying arg1) {
    arg0 = 3.16f16;
    *arg1 = *arg1 * arg0;
}

// Tests function call with float16 arguments.
//; CHECK_HALF: define <{{[0-9]*}} x half> @foo2___vyh(<{{[0-9]*}} x half> %arg0
//; CHECK_HALF: call <{{[0-9]*}} x half> @goo2___vyh(<{{[0-9]*}} x half>
noinline varying float16 goo2(varying float16 arg0) { return arg0 + 7.9; }

varying float16 foo2(varying float16 arg0) { return goo2(arg0); }

// Tests int type -> float16 type conversion.
//; CHECK_HALF: define <{{[0-9]*}} x half> @foo3___vyi(<{{[0-9]*}} x i32>
//; CHECK_HALF: sitofp <{{[0-9]*}} x i32> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to <{{[0-9]*}} x half>
varying float16 foo3(varying int arg0) { return arg0; }

// Tests float16 -> wider float type conversion.
//; CHECK_HALF: define <{{[0-9]*}} x double> @foo4___vyh(<{{[0-9]*}} x half>
//; CHECK_HALF: fpext <{{[0-9]*}} x half> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to <{{[0-9]*}} x double>
varying double foo4(varying float16 arg0) { return arg0; }

// Creates +0, -0, +inf and -inf.
//; CHECK_HALF: define void @foo5
//; CHECK_HALF: store <{{[0-9]*}} x half> zeroinitializer
//; CHECK_HALF: store <{{[0-9]*}} x half> <half 0xH8000
//; CHECK_HALF: store <{{[0-9]*}} x half> <half 0xH7C00
//; CHECK_HALF: store <{{[0-9]*}} x half> <half 0xHFC00
void foo5() {
    float16 pz = +0.f16;
    float16 nz = -0.f16;
    float16 pinf = 0x7c0016;
    float16 ninf = -0xfc0016;
    return;
}

// Testing precedence for float16 with 'higher' precedence type.
//; CHECK_HALF: define <{{[0-9]*}} x i8> @foo6___vyfvyh
//; CHECK_HALF: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = fpext <{{[0-9]*}} x half> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to  <{{[0-9]*}} x float>
//; CHECK_HALF: fadd <{{[0-9]*}} x float>
int8 foo6(float arg0, float16 arg1) {
    int8 arg = arg0 + arg1;
    return arg;
}

// Testing precedence for float16 with 'lower' precedence type.
//; CHECK_HALF: define  <{{[0-9]*}} x i8> @foo7___vysvyh(
//; CHECK_HALF: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = sitofp <{{[0-9]*}} x i16> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to  <{{[0-9]*}} x half>
//; CHECK_HALF: fadd <{{[0-9]*}} x half>
int8 foo7(int16 arg0, float16 arg1) {
    int8 arg = arg0 + arg1;
    return arg;
}
