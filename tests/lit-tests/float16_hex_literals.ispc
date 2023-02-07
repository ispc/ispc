//; RUN: %{ispc} %s --target=avx2-i32x4    -O0 --nostdlib --emit-llvm-text -o - 2>&1 | FileCheck %s
//; RUN: %{ispc} %s --target=avx512knl-x16 -O0 --nostdlib --emit-llvm-text -o - 2>&1 | FileCheck %s
//; RUN: %{ispc} %s --target=avx512skx-x4  -O0 --nostdlib --emit-llvm-text -o - 2>&1 | FileCheck %s
//; RUN: %{ispc} %s --target=neon-i32x4    -O0 --nostdlib --emit-llvm-text -o - 2>&1 | FileCheck %s

//; RUN: not %{ispc} %s --target=sse2-i32x4 -O0 --nostdlib --emit-llvm-text
//; RUN: not %{ispc} %s --target=sse4-i32x4 -O0 --nostdlib --emit-llvm-text
//; RUN: not %{ispc} %s --target=avx1-i32x4 -O0 --nostdlib --emit-llvm-text

// REQUIRES: X86_ENABLED

// CHECK: global half 0xH4000
uniform float16 f1 = 0X1p1f16;
