// Check IR when sizeof is used in global variables initialization.
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap --emit-llvm-text -o - | FileCheck %s --check-prefix=CHECK_IR
// REQUIRES: X86_ENABLED

// CHECK_IR: @test11 = local_unnamed_addr global i32 4
// CHECK_IR: @test12 = local_unnamed_addr global i32 32
// CHECK_IR: @test21 = local_unnamed_addr global i32 4
// CHECK_IR: @test22 = local_unnamed_addr global i32 32
// CHECK_IR: @test31 = local_unnamed_addr global <8 x i32> <i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32>
// CHECK_IR: @test32 = local_unnamed_addr global <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
// CHECK_IR: @test41 = local_unnamed_addr global <8 x i32> <i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32, i32 32>
// CHECK_IR: @test42 = local_unnamed_addr global <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>

uniform int test11 = sizeof(uniform int);
uniform int test12 = sizeof(varying int);
uniform size_t test21 = sizeof(uniform size_t);
uniform size_t test22 = sizeof(varying size_t);

varying int test31 = (varying int)sizeof(varying int);
varying int test32 = (varying int)sizeof(uniform int);
varying size_t test41 = (varying size_t)sizeof(varying size_t);
varying size_t test42 = (varying size_t)sizeof(uniform size_t);