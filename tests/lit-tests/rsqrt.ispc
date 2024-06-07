// RUN: %{ispc} %s --target=host --emit-asm --x86-asm-syntax=intel -O2 -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK:       vrsqrtss
// CHECK:           vrsqrtps
// CHECK-NOT:       vrsqrt14ps

uniform float foo(uniform float x) {
    return rsqrt(x);
}

varying float foo(varying float x) {
    return rsqrt(x);
}
