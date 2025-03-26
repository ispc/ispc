// This test checks that the alignment attribute is correctly applied to
// different ways of declaring structs.

// RUN: %{ispc} --target=host --emit-llvm-text -o - %s 2>&1 | FileCheck %s

struct Sx { } __attribute__((aligned(8)));
// CHECK: @x8 = local_unnamed_addr global %v{{.*}}_uniform_Sx zeroinitializer, align 8
// CHECK: @x4 = local_unnamed_addr global %v{{.*}}_uniform_Sx zeroinitializer, align 4
uniform struct Sx x8;
__attribute__((aligned(4))) uniform struct Sx x4;

// CHECK: @y1024 = local_unnamed_addr global %"v{{.*}}_uniform_$anon0" zeroinitializer, align 1024
// CHECK: @y2048 = local_unnamed_addr global %"v{{.*}}_uniform_$anon1" zeroinitializer, align 2048
uniform struct { int x; } __attribute__((aligned(1024))) y1024;
__attribute__((aligned(2048))) uniform struct { int x; } __attribute__((aligned(1024))) y2048;

// CHECK: @d32 = local_unnamed_addr global %"v{{.*}}_uniform_$anon2" zeroinitializer, align 32
// CHECK: @d64 = local_unnamed_addr global %"v{{.*}}_uniform_$anon3" zeroinitializer, align 64
uniform struct { } __attribute__((aligned(32))) d32;
__attribute__((aligned(64))) uniform struct { } __attribute__((aligned(32))) d64;
