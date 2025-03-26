// This tests checks that the alignment attribute correctly works for global variables.

// RUN: %{ispc} --target=host --nostdlib --emit-llvm-text %s -o - | FileCheck %s

// CHECK: @u = {{.*}}global %v{{.*}}_uniform_U zeroinitializer
uniform struct U { int16 f[3]; } u;
// CHECK: @u4 = {{.*}}global %v{{.*}}_uniform_U4 zeroinitializer, align 4
__attribute__((aligned(4))) uniform struct U4 { int16 f[3]; } u4;
// CHECK: @u8 = {{.*}}global %v{{.*}}_uniform_U8 zeroinitializer, align 8
__attribute__((aligned(8))) uniform struct U8 { int16 f[3]; } u8;
// CHECK: @u16 = {{.*}}global %v{{.*}}_uniform_U16 zeroinitializer, align 16
__attribute__((aligned(16))) uniform struct U16 { int16 f[3]; } u16;

// CHECK: @v = {{.*}}global %v{{.*}}_varying_S zeroinitializer
struct S { int16 f[3]; } v;
// CHECK: @v4 = {{.*}}global %v{{.*}}_varying_S4 zeroinitializer, align 4
__attribute__((aligned(4))) struct S4 { int16 f[3]; } v4;
// CHECK: @v8 = {{.*}}global %v{{.*}}_varying_S8 zeroinitializer, align 8
__attribute__((aligned(8))) struct S8 { int16 f[3]; } v8;
// CHECK: @v{{.*}} = {{.*}}global %v{{.*}}_varying_S16 zeroinitializer, align 16
__attribute__((aligned(16))) struct S16 { int16 f[3]; } v16;


struct G { int8 x; };
// CHECK: @ya_v = {{.*}}global %v{{.*}}_varying_G zeroinitializer, align 8
__attribute__((aligned(8))) struct G ya_v;
// CHECK: @ya_nv = {{.*}}global %v{{.*}}_varying_G zeroinitializer
struct G ya_nv;
// CHECK: @ya = {{.*}}global %v{{.*}}_uniform_G zeroinitializer, align 8
__attribute__((aligned(8))) uniform struct G ya;
// CHECK: @ya_n = {{.*}}global %v{{.*}}_uniform_G zeroinitializer
uniform struct G ya_n;
