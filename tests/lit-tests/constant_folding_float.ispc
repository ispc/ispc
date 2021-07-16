// RUN: %{ispc} %s -O2 --target=host --emit-llvm-text --nostdlib --opt=fast-math -o - | FileCheck %s

// CHECK: @retAddF
// CHECK: ret float 5.000000e+00
uniform float retAddF() {
    return 2.0f + 3.0f;
}

// CHECK: @retSubF
// CHECK: ret float 5.000000e-01
uniform float retSubF() {
    return 3.0f - 2.5f;
}

// CHECK: @retMulF
// CHECK: ret float 7.500000e+00
uniform float retMulF() {
    return 3.0f * 2.5f;
}

// CHECK: @retDivF
// CHECK: ret float 3.000000e+00
uniform float retDivF() {
    return 7.5f / 2.5f;
}

// CHECK: @retNegF
// CHECK: ret float -2.000000e+00
uniform float retNegF() {
    return -2.0f;
}

// CHECK: @retFastMathF
// CHECK: fmul float %{{[a-zA-Z_][a-zA-Z0-9_]*}}, 5.000000e-01
uniform float retFastMathF(uniform float arg) {
    return arg / 2.f;
}

// CHECK: @retLessCompF
// CHECK: ret i1 true
uniform bool retLessCompF() {
    return 2.0f < 3.0f;
}

// CHECK: @retLessEqCompF
// CHECK: ret i1 false
uniform bool retLessEqCompF() {
    return 3.0f <= 2.5f;
}

// CHECK: @retGreatCompF
// CHECK: ret i1 true
uniform bool retGreatCompF() {
    return 3.0f > 2.5f;
}

// CHECK: @retGreatEqCompF
// CHECK: ret i1 true
uniform bool retGreatEqCompF() {
    return 7.5f >= 2.5f;
}

// CHECK: @retEqCompF
// CHECK: ret i1 true
uniform bool retEqCompF() {
    return 2.0f == 2.0f;
}

// CHECK: @retNeqCompF
// CHECK: ret i1 true
uniform bool retNeqCompF() {
    return 2.0f != -2.0f;
}

// CHECK: @retLogAndCompF
// CHECK: ret i1 false
uniform bool retLogAndCompF() {
    return 0.0f && 7.9f;
}

// CHECK: @retLogOrCompF
// CHECK: ret i1 true
uniform bool retLogOrCompF() {
    return 0.0f || 7.9f;
}
