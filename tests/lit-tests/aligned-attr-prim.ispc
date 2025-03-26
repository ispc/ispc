// This tests checks that the alignment attribute correctly works for primitive types

// RUN: %{ispc} --no-discard-value-names --nowrap --target=host --nostdlib --debug-phase=10:10 %s -o t.o | FileCheck %s

// CHECK: @u4 = global i32 0, align 4
// CHECK: @v16 = global <{{.*}} x i32> zeroinitializer, align 16
__attribute__((aligned(4))) uniform int u4;
__attribute__((aligned(16))) varying int v16;

// CHECK: @su256 = global <4 x i32> zeroinitializer, align 256
// CHECK: @sv512 = global [4 x <{{.*}} x i32>] zeroinitializer, align 512
__attribute__((aligned(256))) uniform int<4> su256;
__attribute__((aligned(512))) varying int<4> sv512;

void foo() {
// CHECK: @static.{{.*}}.s_u8 = internal global i32 0, align 8
// CHECK: @static.{{.*}}.s_v32 = internal global <{{.*}} x i32> zeroinitializer, align 32
    __attribute__((aligned(8))) static uniform int s_u8;
    __attribute__((aligned(32))) static varying int s_v32;

    s_u8 = 0;
    s_v32 = 0;

// CHECK: @static.{{.*}}.s_su256 = internal global <{{.*}} x i32> zeroinitializer, align 256
// CHECK: @static.{{.*}}.s_sv512 = internal global [4 x <{{.*}} x i32>] zeroinitializer, align 512
    __attribute__((aligned(256))) static uniform int<4> s_su256;
    __attribute__((aligned(512))) static varying int<4> s_sv512;

    s_su256 = 0;
    s_sv512 = 0;

// CHECK: %s_u64 = alloca i32, align 64
// CHECK: %s_v128 = alloca <{{.*}} x i32>, align 128
    __attribute__((aligned(64))) uniform int s_u64;
    __attribute__((aligned(128))) varying int s_v128;

// CHECK: %su1024 = alloca <{{.*}} x i32>, align 1024
// CHECK: %sv2048 = alloca [4 x <{{.*}} x i32>], align 2048
    __attribute__((aligned(1024))) uniform int<4> su1024;
    __attribute__((aligned(2048))) varying int<4> sv2048;
}
