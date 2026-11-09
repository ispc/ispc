// Verifies the layout of null and zero varying SOA pointer initializers on a
// 32-bit target: the pointer and the offset are both smeared to <8 x i32>.

// RUN: %{ispc} %s --target=avx2-i32x8 --arch=x86 --nowrap -O0 --emit-llvm-text -o - | FileCheck %s

// macOS doesn't support 32-bit x86.
// REQUIRES: X86_ENABLED && !MACOS_HOST

struct P3 { float x, y, z; };

// CHECK: @gnull = global { <8 x i32>, <8 x i32> } zeroinitializer
// CHECK: @gzero = global { <8 x i32>, <8 x i32> } zeroinitializer
// CHECK: @garr = global [2 x { <8 x i32>, <8 x i32> }] zeroinitializer
soa<8> P3 * varying gnull = NULL;
soa<8> P3 * varying gzero = 0;
soa<8> P3 * varying garr[2] = {NULL, 0};
