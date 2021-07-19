//; RUN: %{ispc} %s --target=host --emit-llvm-text -O0 -o - | FileCheck %s

// Tests float16 constant literals.
//; CHECK: define void @foo_float16
//; CHECK: store half 0xH409A
//; CHECK: store half 0xH409A
//; CHECK: store half 0xH5A40
//; CHECK: store half 0xH4800
//; CHECK: store half 0xH5A40
//; CHECK: store half 0xH3000
//; CHECK: store half 0xH0000
//; CHECK: store half 0xH8000
//; CHECK: store half 0xH7C00
//; CHECK: store half 0xHFC00
//; CHECK: store half 0xH0002
void foo_float16() {
    uniform float16 f16val1 = 2.3f16; // f16 suffix.
    uniform float16 f16val2 = 2.3F16; // F16 suffix.
    uniform float16 f16val3 = 2e+2f16; // exponential form with signed exponential.
    uniform float16 f16val4 = 0x1.p3f16; // normalized hex form with no sign.
    uniform float16 f16val5 = 2e2f16; //exponential form with no sign for exponential.
    uniform float16 f16val6 = 0x1.p-3f16; // normalized hex form with sign.
    uniform float16 f16val7 = 0x0.p+1f16; // +0
    uniform float16 f16val8 = -0x0.p+1f16; // -0
    uniform float16 f16val9 = +0x1.9p+308f16; // +infinity
    uniform float16 f16val10 = -0x1.9p+308f16; // -infinity
    uniform float16 f16val11 = 0x1p-23f16; // denormal
}

// Tests float constant literals.
//; CHECK: define void @foo_float32
//; CHECK: store float 0x4002666660000000
//; CHECK: store float 0x4002666660000000
//; CHECK: store float 0x4002666660000000
//; CHECK: store float 2.000000e+02
//; CHECK: store float 8.000000e+00
//; CHECK: store float 2.000000e+02
//; CHECK: store float 1.250000e-01
//; CHECK: store float 0.000000e+00
//; CHECK: store float -0.000000e+00
//; CHECK: store float 0x7FF0000000000000
//; CHECK: store float 0xFFF0000000000000
//; CHECK: store float 0x36A0000000000000
void foo_float32() {
    uniform float f32val0 = 2.3; // no suffix.
    uniform float f32val1 = 2.3f;  // f suffix.
    uniform float f32val2 = 2.3F; // F suffix
    uniform float f32val3 = 2e+2f; // exponential form with signed exponential.
    uniform float f32val4 = 0x1.p3f; // normalized hex form with no sign.
    uniform float f32val5 = 2e2f; //exponential form with no sign for exponential.
    uniform float f32val6 = 0x1.p-3f; // normalized hex form with sign.
    uniform float f32val7 = 0x0.p+1f; // +0
    uniform float f32val8 = -0x0.p+1f; // -0
    uniform float f32val9 = +0x1.9p+308f; // +infinity
    uniform float f32val10 = -0x1.9p+308f; // -infinity
    uniform float f32val11 = 0x1p-149f; // denormal
}

// Tests double constant literals.
//; CHECK: define void @foo_double
//; CHECK: store double 2.300000e+00
//; CHECK: store double 2.300000e+00
//; CHECK: store double 2.000000e+02
//; CHECK: store double 8.000000e+00
//; CHECK: store double 2.000000e+02
//; CHECK: store double 1.250000e-01
//; CHECK: store double 0.000000e+00
//; CHECK: store double -0.000000e+00
//; CHECK: store double 0x7FF0000000000000
//; CHECK: store double 0xFFF0000000000000
//; CHECK: store double 1.401300e-45
void foo_double() {
    uniform double dval1 = 2.3d; // d suffix.
    uniform double dval2 = 2.3D; // D suffix
    uniform double dval3 = 2e+2d; // exponential form with signed exponential.
    uniform double dval4 = 0x1.p3d; // normalized hex form with no sign.
    uniform double dval5 = 2e2d; //exponential form with no sign for exponential.
    uniform double dval6 = 0x1.p-3d; // normalized hex form with sign.
    uniform double dval7 = 0x0.p+1d; // +0
    uniform double dval8 = -0x0.p+1d; // -0
    uniform double dval9 = +0x1.9p+8608d; // +infinity
    uniform double dval10 = -0x1.9p+8608d; // -infinity
    uniform double dval11 = 0x1.00001262d4ab6p-149d; // denormal
}
