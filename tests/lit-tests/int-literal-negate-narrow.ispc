// Negating an unsuffixed integer literal must not wrap in the narrow literal
// type chosen for targets with 8-bit and 16-bit data types.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i8x16 -O2 --emit-llvm-text --nowrap -o - | FileCheck %s
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i16x8 -O2 --emit-llvm-text --nowrap -o - | FileCheck %s
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 -O2 --emit-llvm-text --nowrap -o - | FileCheck %s

// REQUIRES: ARM_ENABLED

// CHECK-LABEL: define {{.*}}i32 @neg128(
// CHECK: ret i32 -128
export uniform int neg128() { return -128; }

// CHECK-LABEL: define {{.*}}i32 @neg130(
// CHECK: ret i32 -130
export uniform int neg130() { return -130; }

// CHECK-LABEL: define {{.*}}i32 @neg255(
// CHECK: ret i32 -255
export uniform int neg255() { return -255; }

// CHECK-LABEL: define {{.*}}i32 @neg32768(
// CHECK: ret i32 -32768
export uniform int neg32768() { return -32768; }

// CHECK-LABEL: define {{.*}}i32 @neg40000(
// CHECK: ret i32 -40000
export uniform int neg40000() { return -40000; }
