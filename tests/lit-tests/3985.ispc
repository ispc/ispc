// Regression test for https://github.com/ispc/ispc/issues/3985
// With 32-bit addressing, the byte offset of an unsigned index is computed in
// 32 bits and then zero-extended, so the multiply must be nuw, not nsw (nsw
// would make offsets above INT32_MAX poison). Signed indices keep nsw.

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O0 --emit-llvm-text -o - | FileCheck %s
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O0 --emit-llvm-text --wrap-signed-int -o - | FileCheck %s --check-prefix=CHECK_WRAP
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O0 --emit-llvm-text --addressing=64 -o - | FileCheck %s --check-prefix=CHECK_A64

// REQUIRES: X86_ENABLED

// CHECK_A64-LABEL: define void @gather_unsigned
// CHECK_A64: [[IDX:%.*]] = zext <8 x i32> %i_load to <8 x i64>
// CHECK_A64: mul nuw <8 x i64> splat (i64 4), [[IDX]]
// CHECK_A64-LABEL: define void @gather_signed
// CHECK_A64: [[IDX:%.*]] = sext <8 x i32> %i_load to <8 x i64>
// CHECK_A64: mul nsw <8 x i64> splat (i64 4), [[IDX]]

// CHECK-LABEL: define void @gather_unsigned
// CHECK: [[OFF:%.*]] = mul nuw <8 x i32> splat (i32 4), %i_load
// CHECK: zext <8 x i32> [[OFF]] to <8 x i64>
// CHECK_WRAP-LABEL: define void @gather_unsigned
// CHECK_WRAP: [[OFF:%.*]] = mul <8 x i32> splat (i32 4), %i_load
// CHECK_WRAP: zext <8 x i32> [[OFF]] to <8 x i64>
void gather_unsigned(uniform float p[], uniform unsigned int32 idx[], uniform float r[]) {
    unsigned int32 i = idx[programIndex];
    r[programIndex] = p[i];
}

// CHECK-LABEL: define void @varying_ptr_uniform_unsigned
// CHECK: mul nuw i32 4,
// CHECK: zext <8 x i32> {{.*}} to <8 x i64>
void varying_ptr_uniform_unsigned(uniform float *varying p, uniform unsigned int32 i, uniform float r[]) {
    r[programIndex] = p[i];
}

// CHECK-LABEL: define void @gather_signed
// CHECK: [[OFF:%.*]] = mul nsw <8 x i32> splat (i32 4), %i_load
// CHECK: sext <8 x i32> [[OFF]] to <8 x i64>
void gather_signed(uniform float p[], uniform int32 idx[], uniform float r[]) {
    int32 i = idx[programIndex];
    r[programIndex] = p[i];
}
