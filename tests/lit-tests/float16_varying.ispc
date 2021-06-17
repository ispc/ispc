//; RUN: %{ispc} %s --target=host --emit-llvm-text -O0 -o - | FileCheck %s

// Tests float16 constant parsing in *e*f16 form.
//; CHECK: define <{{[0-9]*}} x half> @foo0___vyh(<{{[0-9]*}} x half>
//; CHECK: fadd <{{[0-9]*}} x half> %{{[a-zA-Z_][a-zA-Z0-9_]*}}, <half 0xH011D, half 0xH011D
varying float16 foo0(varying float16 arg0) {
    arg0 = arg0 + 1.7e-5f16;
    return arg0;
}

// Tests float16 constant parsing in *.*f16 form.
//; CHECK: define void @foo1___REFvyhvy_3C_vyh_3E_(<{{[0-9]*}} x half>* noalias %arg0, <{{[0-9]*}} x i64> %arg1
//; CHECK: fmul <{{[0-9]*}} x half>
void foo1(varying float16 &arg0, varying float16 *varying arg1) {
    arg0 = 3.16f16;
    *arg1 = *arg1 * arg0;
}

// Tests function call with float16 arguments.
//; CHECK: define <{{[0-9]*}} x half> @foo2___vyh(<{{[0-9]*}} x half> %arg0
//; CHECK: call <{{[0-9]*}} x half> @goo2___vyh(<{{[0-9]*}} x half>
noinline varying float16 goo2(varying float16 arg0) { return arg0 + 7.9; }

varying float16 foo2(varying float16 arg0) { return goo2(arg0); }

// Tests int type -> float16 type conversion.
//; CHECK: define <{{[0-9]*}} x half> @foo3___vyi(<{{[0-9]*}} x i32>
//; CHECK: sitofp <{{[0-9]*}} x i32> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to <{{[0-9]*}} x half>
varying float16 foo3(varying int arg0) { return arg0; }

// Tests float16 -> wider float type conversion.
//; CHECK: define <{{[0-9]*}} x double> @foo4___vyh(<{{[0-9]*}} x half>
//; CHECK: fpext <{{[0-9]*}} x half> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to <{{[0-9]*}} x double>
varying double foo4(varying float16 arg0) { return arg0; }

// Creates +0, -0, +inf and -inf.
//; CHECK: define void @foo5
//; CHECK: store <{{[0-9]*}} x half> zeroinitializer
//; CHECK: store <{{[0-9]*}} x half> <half 0xH8000
//; CHECK: store <{{[0-9]*}} x half> <half 0xH7C00
//; CHECK: store <{{[0-9]*}} x half> <half 0xHFC00
void foo5() {
    float16 pz = +0.f16;
    float16 nz = -0.f16;
    float16 pinf = 0x7c0016;
    float16 ninf = -0xfc0016;
    return;
}

// Testing precedence for float16 with 'higher' precedence type.
//; CHECK: define <{{[0-9]*}} x i8> @foo6___vyfvyh
//; CHECK: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = fpext <{{[0-9]*}} x half> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to  <{{[0-9]*}} x float>
//; CHECK: fadd <{{[0-9]*}} x float>
int8 foo6(float arg0, float16 arg1) {
    int8 arg = arg0 + arg1;
    return arg;
}

// Testing precedence for float16 with 'lower' precedence type.
//; CHECK: define  <{{[0-9]*}} x i8> @foo7___vysvyh(
//; CHECK: %{{[a-zA-Z_][a-zA-Z0-9_]*}} = sitofp <{{[0-9]*}} x i16> %{{[a-zA-Z_][a-zA-Z0-9_]*}} to  <{{[0-9]*}} x half>
//; CHECK: fadd <{{[0-9]*}} x half>
int8 foo7(int16 arg0, float16 arg1) {
    int8 arg = arg0 + arg1;
    return arg;
}
