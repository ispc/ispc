// RUN: %{ispc} %s -O2 --target=host --emit-llvm-text --nostdlib --opt=fast-math -o - | FileCheck %s

// CHECK: @retAddD
// CHECK: ret double 5.000000e+00
uniform double retAddD() {
    return 2.0d + 3.0d;
}

// CHECK: @retSubD
// CHECK: ret double 5.000000e-01
uniform double retSubD() {
    return 3.0d - 2.5d;
}

// CHECK: @retMulD
// CHECK: ret double 7.500000e+00
uniform double retMulD() {
    return 3.0d * 2.5d;
}

// CHECK: @retDivD
// CHECK: ret double 3.000000e+00
uniform double retDivD() {
    return 7.5d / 2.5d;
}

// CHECK: @retNegD
// CHECK: ret double -2.000000e+00
uniform double retNegD() {
    return -2.0d;
}

// CHECK: @retFastMathD
// CHECK: fmul double %{{[a-zA-Z_][a-zA-Z0-9_]*}}, 5.000000e-01
uniform double retFastMathD(uniform double arg) {
    return arg / 2.d;
}

// CHECK: @retLessCompD
// CHECK: ret i1 true
uniform bool retLessCompD() {
    return 2.0d < 3.0d;
}

// CHECK: @retLessEqCompD
// CHECK: ret i1 false
uniform bool retLessEqCompD() {
    return 3.0d <= 2.5d;
}

// CHECK: @retGreatCompD
// CHECK: ret i1 true
uniform bool retGreatCompD() {
    return 3.0d > 2.5d;
}

// CHECK: @retGreatEqCompD
// CHECK: ret i1 true
uniform bool retGreatEqCompD() {
    return 7.5d >= 2.5d;
}

// CHECK: @retEqCompD
// CHECK: ret i1 true
uniform bool retEqCompD() {
    return 2.0d == 2.0d;
}

// CHECK: @retNeqCompD
// CHECK: ret i1 true
uniform bool retNeqCompD() {
    return 2.0d != -2.0d;
}

// CHECK: @retLogAndCompD
// CHECK: ret i1 false
uniform bool retLogAndCompD() {
    return 0.0d && 7.9d;
}

// CHECK: @retLogOrCompD
// CHECK: ret i1 true
uniform bool retLogOrCompD() {
    return 0.0d || 7.9d;
}
