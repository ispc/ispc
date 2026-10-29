// Regression test for https://github.com/ispc/ispc/issues/3985
// With 32-bit addressing, ImproveMemoryOps narrows 64-bit gather and scatter
// offsets to 32 bits. The 32-bit variants and the passes after it sign extend
// them, so a zero extended offset (of an unsigned index) stays 64-bit, and
// zext(c + v) may only be split into zext(c) + zext(v) if the add doesn't
// wrap. 3985-1.ispc runs these cases.

// RUN: %{ispc} %s --target=sse4-i32x4 -O2 --nostdlib --nowrap --woff --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 -O2 --nostdlib --nowrap --woff --emit-llvm-text -o - | FileCheck %s --check-prefix=AVX2
// RUN: %{ispc} %s --target=sse4-i32x4 -O2 --nostdlib --nowrap --woff --addressing=64 --emit-llvm-text -o - | FileCheck %s --check-prefix=A64
// RUN: %{ispc} %s --target=avx2-i32x8 -O2 --nostdlib --nowrap --woff --addressing=64 --emit-llvm-text -o - | FileCheck %s --check-prefix=A64

// REQUIRES: X86_ENABLED

// p[i]: the shl nuw by 2 is pulled out as the scale of the 64-bit offsets.
// CHECK-LABEL: @gather_u___
// CHECK: [[I:%.*]] = zext <4 x i32> {{%.*}} to <4 x i64>
// CHECK: [[LANE:%.*]] = extractelement <4 x i64> [[I]], {{i32|i64}} 0
// CHECK: shl nuw nsw i64 [[LANE]], 2
// CHECK-NOT: sext i32
// CHECK: {{^}}}
void gather_u(uniform float p[], uniform unsigned int32 idx[], uniform float r[]) {
    unsigned int32 i = idx[programIndex];
    r[programIndex] = p[i];
}

// p[i - 1]: the 32-bit offset i * 4 - 4 wraps for every i, so it is neither
// split nor narrowed. Its lanes are zero extended.
// CHECK-LABEL: @gather_u_minus_1___
// CHECK: zext <4 x i32> {{%.*}} to <4 x i64>
// CHECK-NOT: sext i32
// CHECK: {{^}}}
void gather_u_minus_1(uniform float p[], uniform unsigned int32 idx[], uniform float r[]) {
    unsigned int32 i = idx[programIndex];
    r[programIndex] = p[i - 1];
}

// A contiguous store through an unsigned index: the scatter becomes a vector
// store, whose offset stays zero extended. base + <0, 1, ...> may wrap, so the
// address is computed from the first lane that is on.
// CHECK-LABEL: @store_u___
// CHECK: call {{.*}} @llvm.cttz.i4
// CHECK: zext i32 {{%.*}} to i64
// CHECK-NOT: sext i32
// CHECK: {{^}}}
// AVX2-LABEL: @store_u___
// AVX2: call {{.*}} @llvm.cttz.i8
// AVX2: zext i32 {{%.*}} to i64
// AVX2-NOT: sext i32
// AVX2: {{^}}}
void store_u(uniform float p[], uniform unsigned int32 base) { p[base + programIndex] = 3.0f; }

// p[j - 1] in "if (j > 0)": the lane constants <-1, 0, ...> make the offset of
// lane 0 wrap for j = programIndex while it is off (see 3985-1.ispc), so the
// address is computed from the first lane that is on.
// CHECK-LABEL: @store_u_minus_1___
// CHECK: call {{.*}} @llvm.cttz.i4
// CHECK: zext i32 {{%.*}} to i64
// CHECK-NOT: sext i32
// CHECK: {{^}}}
// AVX2-LABEL: @store_u_minus_1___
// AVX2: call {{.*}} @llvm.cttz.i8
// AVX2: zext i32 {{%.*}} to i64
// AVX2-NOT: sext i32
// AVX2: {{^}}}
void store_u_minus_1(uniform float p[], uniform unsigned int32 base) {
    unsigned int32 j = base + programIndex;
    if (j > 0) {
        p[j - 1] = 3.0f;
    }
}

// A varying unsigned loop index, incremented without nuw, may wrap: the first
// lane that is on gives the address.
// CHECK-LABEL: @loop_u___
// CHECK: call {{.*}} @llvm.cttz.i4
// CHECK: {{^}}}
// AVX2-LABEL: @loop_u___
// AVX2: call {{.*}} @llvm.cttz.i8
// AVX2: {{^}}}
void loop_u(uniform float a[], uniform float r[], uniform unsigned int32 n) {
    float s = 0;
    for (unsigned int32 i = programIndex; i < n; i += programCount) {
        s += a[i];
    }
    r[programIndex] = s;
}

// With all lanes on, the address is computed from the first lane.
// CHECK-LABEL: @store_u_all_on___
// CHECK-NOT: @llvm.cttz
// CHECK: zext i32 %base to i64
// CHECK-NOT: sext i32
// CHECK: {{^}}}
// AVX2-LABEL: @store_u_all_on___
// AVX2-NOT: @llvm.cttz
// AVX2: zext i32 %base to i64
// AVX2-NOT: sext i32
// AVX2: {{^}}}
void store_u_all_on(uniform float p[], uniform unsigned int32 base) {
    foreach (k = 0 ... programCount) {
        p[base + k] = 3.0f;
    }
}

// With 64-bit addressing, InstCombine turns the offset of a signed loop counter
// into zext(shl nuw (or disjoint %i, <0, 1, ...>), 1), where %i is a phi and the
// constant part is split off unfolded. Neither may wrap, so the masked accesses
// use the address of lane 0 instead of looking for the first lane that is on.
// A64-LABEL: @masked_counter___
// A64-NOT: @llvm.cttz
// A64: @llvm.masked.load.v{{4|8}}i16
// A64-NOT: @llvm.cttz
// A64: {{^}}}
void masked_counter(uniform int16 a[], uniform int16 b[], uniform int n, uniform bool m[]) {
    bool mask = m[programIndex];
    for (int i = 0; i < n; i += programCount) {
        if (mask) {
            b[i + programIndex] = a[i + programIndex];
        }
    }
}

// A narrow unsigned index wraps in 8 or 16 bits: lane 0 of (uint8)(j - 1) is
// 255 for b = 0 while it is off, so the address is computed from the first
// lane that is on.
// CHECK-LABEL: @load_u8_minus_1___
// CHECK: call {{.*}} @llvm.cttz.i4
// CHECK: {{^}}}
// AVX2-LABEL: @load_u8_minus_1___
// AVX2: call {{.*}} @llvm.cttz.i8
// AVX2: {{^}}}
void load_u8_minus_1(uniform float r[], uniform float p[], uniform uint8 b) {
    uint8 j = b + (uint8)programIndex;
    float v = -1;
    if (j > 0) {
        v = p[(uint8)(j - 1)];
    }
    r[programIndex] = v;
}

// CHECK-LABEL: @store_u16_minus_1___
// CHECK: call {{.*}} @llvm.cttz.i4
// CHECK: {{^}}}
// AVX2-LABEL: @store_u16_minus_1___
// AVX2: call {{.*}} @llvm.cttz.i8
// AVX2: {{^}}}
void store_u16_minus_1(uniform float p[], uniform uint16 b) {
    uint16 j = b + (uint16)programIndex;
    if (j > 0) {
        p[(uint16)(j - 1)] = 1.0f;
    }
}

// The constant part of the offset, 2^31, doesn't fit in the 32-bit variants,
// which sign extend it: the offsets stay 64-bit.
// CHECK-LABEL: @gather_large_const___
// CHECK: getelementptr i8, ptr {{%.*}}, i64 2147483648
// CHECK-NOT: i64 -2147483648
// CHECK: {{^}}}
// AVX2-LABEL: @gather_large_const___
// AVX2-NOT: @llvm.x86.avx2.gather.d.ps
// AVX2: @llvm.x86.avx2.gather.q.ps.256
// AVX2-NOT: @llvm.x86.avx2.gather.d.ps
// AVX2: {{^}}}
void gather_large_const(uniform float r[], uniform float p[], uniform uint16 j[]) {
    foreach (k = 0 ... programCount) {
        r[k] = p[(unsigned int32)j[k] + 0x20000000u];
    }
}

// Also if the whole offset is a constant.
// CHECK-LABEL: @store_large_const___
// CHECK: getelementptr i8, ptr %p, i64 2147483648
// CHECK-NOT: i64 -2147483648
// CHECK: {{^}}}
// AVX2-LABEL: @store_large_const___
// AVX2: getelementptr i8, ptr %p, i64 2147483648
// AVX2-NOT: i64 -2147483648
// AVX2: {{^}}}
void store_large_const(uniform float p[]) {
    unsigned int32 i = 0x20000000u + 2 * programIndex;
    p[i] = 1.0f;
}
