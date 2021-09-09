// RUN: %{ispc} %s --target=avx2-i32x8 --cpu=znver2 --print-target --arch=x86-64 --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --cpu=ps5 --print-target --arch=x86-64 --nostdlib | FileCheck %s

// RUN: %{ispc} %s --target=sse2-i32x4 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=sse2-i32x8 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=sse4-i32x4 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=sse4-i32x8 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=sse4-i16x8 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=sse4-i8x16 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x4 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x8 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i32x16 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx1-i64x4 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i8x32 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i16x16 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x4 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x16 --cpu=ps5 --print-target --nostdlib | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i64x4 --cpu=ps5 --print-target --nostdlib | FileCheck %s


// REQUIRES: X86_ENABLED

// CHECK: Target CPU: znver2
int i;
