// This tests checks that the alignment attribute correctly works with typedefs.

// RUN: %{ispc} --no-discard-value-names --nowrap --target=host --nostdlib --debug-phase=10:10 %s -o t.o | FileCheck %s

typedef uniform int __attribute__((aligned(8))) aligned_int;

// CHECK: @x = global i32 0, align 8
aligned_int x;

typedef struct S { int x; } __attribute__((aligned(16))) aligned_struct;

// CHECK: @y = global %v{{.*}}_varying_S zeroinitializer, align 16
aligned_struct y;

typedef struct X { int x; int y; } __attribute__((aligned(32))) Point, *PointPtr, my_point;

// CHECK: @m1 = global %v{{.*}}_varying_X zeroinitializer, align 32
// CHECK: @m2 = global %v{{.*}}_varying_X zeroinitializer, align 32
// CHECK: @mp = global ptr null
Point m1;
my_point m2;
uniform PointPtr mp;

typedef struct { int x; int y; } __attribute__((aligned(64))) Point2, *PointPtr2, my_point2;

// CHECK: @m1_1 = global %v{{.*}}_varying_Point2 zeroinitializer, align 64
// CHECK: @m2_1 = global %v{{.*}}_varying_my_point2 zeroinitializer, align 64
// CHECK: @m2p = global ptr null
Point2 m1_1;
my_point2 m2_1;
uniform PointPtr2 m2p;
