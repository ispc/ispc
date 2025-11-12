// RUN: %{ispc} %s --target=sse4.1-i8x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=sse4.2-i8x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i16x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2vnni-i32x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512icl-x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-i32x16 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx512skx-x16 --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// It's expected that the shuffle call will generate a shufflevector
// instruction in the final LLVM IR after all optimization passes, with
// the permutation indices preserved exactly as specified in the source code.
// No undef/poison values should be needed here.

// CHECK: shufflevector <16 x i32> {{%.*}}, <16 x i32> {{%.*}}, <16 x i32> <i32 0, i32 6, i32 1, i32 7, i32 4, i32 0, i32 5, i32 1, i32 8, i32 4, i32 9, i32 25, i32 2, i32 28, i32 3, i32 29>

// CHECK-NOT: poison
// CHECK-NOT: undef

float foo(int d[]) {
  const int p = {0, 6, 1, 7, 4, 0, 5, 1, 8, 4, 9, 25, 2, 28, 3, 29};
  return shuffle(d[0], d[1], p);
}
