// Test to check 'alloca()' implementation.
// RUN: %{ispc} %s --target=host --nowrap -O0 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=gen9-x8 --nowrap -O0 --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECK-XE

// REQUIRES: X86_ENABLED
// REQUIRES: XE_ENABLED

extern void goo(uniform int *uniform alloca_ptr);
// CHECK: @foo0
// CHECK: alloca i8, i32 8, align 16
// CHECK-XE: @foo0
// CHECK-XE: alloca i8, i64 8, align 16
extern void foo0() {
    goo((uniform int *uniform)alloca(8));
}

// CHECK: @foo1
// CHECK: alloca i8, i32 %size_load, align 16
// CHECK-XE: @foo1
// CHECK-XE: alloca i8, i64 %size, align 16
extern void foo1(uniform size_t size) {
    goo((uniform int *uniform)alloca(size));
}

// CHECK: @foo2
// CHECK:  alloca i8, i32 %size_load_to_uint32, align 16
// CHECK-XE: @foo2
// CHECK-XE:  alloca i8, i64 %size_load_to_uint64, align 16
extern void foo2(uniform int8 size) {
    goo((uniform int *uniform)alloca(size));
}

// CHECK: @foo3
// CHECK:  alloca i8, i32 %size_load_to_uint32, align 16
// CHECK-XE: @foo3
// CHECK-XE:  alloca i8, i64 %size_load_to_uint64, align 16
extern void foo3(uniform float size) {
    goo((uniform int *uniform)alloca(size));
}

// CHECK: @foo4
// CHECK: alloca i8, i32 %size_load_load_to_uint32, align 16
// CHECK-XE: @foo4
// CHECK-XE: alloca i8, i64 %size_load_load_to_uint64, align 16
extern void foo4(uniform float *uniform size) {
    goo((uniform int *uniform)alloca(*size));
}

// CHECK: @foo5
// CHECK: alloca i8, i32 8, align 16
// CHECK-XE: @foo5
// CHECK-XE: alloca i8, i64 8, align 16
extern void foo5() {
    goo((uniform int *uniform)alloca(sizeof(uniform double)));
}
