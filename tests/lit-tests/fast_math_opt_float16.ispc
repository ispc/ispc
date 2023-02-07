// RUN: %{ispc} %s --target=sse2-i32x4 --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV
// RUN: %{ispc} %s --target=sse4-i32x4 --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV
// RUN: %{ispc} %s --target=avx1-i32x4 --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV
// RUN: %{ispc} %s --target=avx2-i32x4 --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV
// RUN: %{ispc} %s --target=avx512knl-x16 --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV
// RUN: %{ispc} %s --target=avx512skx-x4 --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV
// RUN: %{ispc} %s --target=neon-i32x4 --opt=fast-math -O0 --emit-llvm-text -o -| FileCheck %s -check-prefix=CHECK_NOFDIV

// RUN: not %{ispc} %s --target=sse2-i32x4 --opt=fast-math --nostdlib --nowrap
// RUN: not %{ispc} %s --target=sse4-i32x4 --opt=fast-math --nostdlib --nowrap
// RUN: not %{ispc} %s --target=avx1-i32x4 --opt=fast-math --nostdlib --nowrap

// REQUIRES: X86_ENABLED

// CHECK_NOFDIV-NOT: fdiv

float16 f1(float16 a) {
  return a/2.f16;
}

float16 f2(float16 a, float16 b) {
  return a/b;
}
