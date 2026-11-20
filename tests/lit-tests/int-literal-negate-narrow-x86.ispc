// Negating an unsuffixed integer literal must not wrap in the narrow literal
// type chosen for targets with 8-bit and 16-bit data types. The same checks as
// int-literal-negate-narrow.ispc, on x86 targets.
// RUN: %{ispc} %s --arch=x86-64 --target=sse4.2-i8x16 -O2 --emit-llvm-text --nowrap -o - | FileCheck %s --check-prefixes=CHECK,I8
// RUN: %{ispc} %s --arch=x86-64 --target=sse4.2-i16x8 -O2 --emit-llvm-text --nowrap -o - | FileCheck %s --check-prefixes=CHECK,I16
// RUN: %{ispc} %s --arch=x86-64 --target=sse4.2-i32x4 -O2 --emit-llvm-text --nowrap -o - | FileCheck %s --check-prefixes=CHECK,I32

// REQUIRES: X86_ENABLED

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

// This multiplication uses the type of the positive literal to determine its width.
static uniform int scale(uniform uint16 x) { return x * 300; }

// CHECK-LABEL: define {{.*}}i32 @positiveProduct(
// I8: ret i32 90000
// I16: ret i32 24464
// I32: ret i32 90000
export uniform int positiveProduct() { return scale(300); }

// CHECK-LABEL: define {{.*}}i32 @positiveComparison(
// I8: ret i32 1
// I16: ret i32 0
// I32: ret i32 0
export uniform int positiveComparison() {
    uniform int8 x = -1;
    return x == 0xff;
}

static uniform int tag(uniform int8 x) { return 1; }
static uniform int tag(uniform uint8 x) { return 2; }
static uniform int tag(uniform int16 x) { return 3; }
static uniform int tag(uniform uint16 x) { return 4; }
static uniform int tag(uniform int32 x) { return 5; }

// CHECK-LABEL: define {{.*}}i32 @positiveOverloads(
// I8: ret i32 255
// I16: ret i32 334
// I32: ret i32 555
export uniform int positiveOverloads() { return 100 * tag(200) + 10 * tag(300) + tag(40000); }

// CHECK-LABEL: define {{.*}}i32 @negativeOverloads(
// I8: ret i32 1555
// I16: ret i32 3335
// I32: ret i32 5555
export uniform int negativeOverloads() {
    return 1000 * tag(-128) + 100 * tag(-200) + 10 * tag(-32768) + tag(-32769);
}

// CHECK-LABEL: define {{.*}}i64 @unsignedSuffix(
// CHECK: ret i64 4294967096
export uniform uint64 unsignedSuffix() { return -200u; }

// CHECK-LABEL: define {{.*}}i32 @unsignedCast(
// CHECK: ret i32 56
export uniform int unsignedCast() { return -(uniform uint8)200; }

// CHECK-LABEL: define {{.*}}i32 @otherForms(
// CHECK: ret i32 -511
export uniform int otherForms() { return -0xff + -(128) + -+128; }

static uniform int pick(uniform int x) {
    switch (x) {
    case -128:
        return 1;
    case -200:
        return 2;
    default:
        return 0;
    }
}

// CHECK-LABEL: define {{.*}}i32 @switchCases(
// CHECK: ret i32 12
export uniform int switchCases() { return 10 * pick(-128) + pick(-200); }
