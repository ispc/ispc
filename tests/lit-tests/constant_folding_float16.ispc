// RUN: %{ispc} %s -O2 --target=avx2-i32x4 --emit-llvm-text --nostdlib --opt=fast-math -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --target=avx512knl-x16 --emit-llvm-text --nostdlib --opt=fast-math -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --target=avx512skx-x4 --emit-llvm-text --nostdlib --opt=fast-math -o - | FileCheck %s
// RUN: %{ispc} %s -O2 --target=neon-i32x4 --emit-llvm-text --nostdlib --opt=fast-math -o - | FileCheck %s

// RUN: not %{ispc} %s -O2 --target=sse2-i32x4 --emit-llvm-text --nostdlib --opt=fast-math -o %t
// RUN: not %{ispc} %s -O2 --target=sse4-i32x4 --emit-llvm-text --nostdlib --opt=fast-math -o %t
// RUN: not %{ispc} %s -O2 --target=avx1-i32x4 --emit-llvm-text --nostdlib --opt=fast-math -o %t

// REQUIRES: X86_ENABLED

// CHECK: @retAddF16
// CHECK: ret half 0xH4500
uniform float16 retAddF16() {
    return 2.0f16 + 3.0f16;
}

// CHECK: @retSubF16
// CHECK: ret half 0xH3800
uniform float16 retSubF16() {
    return 3.0f16 - 2.5f16;
}

// CHECK: @retMulF16
// CHECK: ret half 0xH4780
uniform float16 retMulF16() {
    return 3.0f16 * 2.5f16;
}

// CHECK: @retDivF16
// CHECK: ret half 0xH4200
uniform float16 retDivF16() {
    return 7.5f16 / 2.5f16;
}

// CHECK: @retNegF16
// CHECK: ret half 0xHC000
uniform float16 retNegF16() {
    return -2.0f16;
}

// CHECK: @retFastMathF16
// CHECK: fmul half %{{[a-zA-Z_][a-zA-Z0-9_]*}}, 0xH3800
uniform float16 retFastMathF16(uniform float16 arg) {
    return arg / 2.f16;
}

// CHECK: @retLessCompF16
// CHECK: ret i1 true
uniform bool retLessCompF16() {
    return 2.0f16 < 3.0f16;
}

// CHECK: @retLessEqCompF16
// CHECK: ret i1 false
uniform bool retLessEqCompF16() {
    return 3.0f16 <= 2.5f16;
}

// CHECK: @retGreatCompF16
// CHECK: ret i1 true
uniform bool retGreatCompF16() {
    return 3.0f16 > 2.5f16;
}

// CHECK: @retGreatEqCompF16
// CHECK: ret i1 true
uniform bool retGreatEqCompF16() {
    return 7.5f16 >= 2.5f16;
}

// CHECK: @retEqCompF16
// CHECK: ret i1 true
uniform bool retEqCompF16() {
    return 2.0f16 == 2.0f16;
}

// CHECK: @retNeqCompF16
// CHECK: ret i1 true
uniform bool retNeqCompF16() {
    return 2.0f16 != -2.0f16;
}

// CHECK: @retLogAndCompF16
// CHECK: ret i1 false
uniform bool retLogAndCompF16() {
    return 0.0f16 && 7.9f16;
}

// CHECK: @retLogOrCompF16
// CHECK: ret i1 true
uniform bool retLogOrCompF16() {
    return 0.0f16 || 7.9f16;
}
