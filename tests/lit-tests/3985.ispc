// Regression test for https://github.com/ispc/ispc/issues/3985
// With 32-bit addressing, the byte offset of an unsigned index is computed in
// 32 bits and then zero-extended, so the multiply must be nuw, not nsw (nsw
// would make offsets above INT32_MAX poison). Signed indices keep nsw.
// --wrap-signed-int turns off nuw, too. A full-width unsigned offset (uint64
// with --addressing=64, or a 32-bit target, see 3985-4.ispc) has no flag: it
// may wrap, like the pointer it is added to.

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
// CHECK_A64-LABEL: define void @gather_unsigned64
// CHECK_A64: %mul__i_load = mul <8 x i64> splat (i64 4), %i_load

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

// A uniform index of a varying pointer: the offset is computed once and then
// smeared.
// CHECK-LABEL: define void @varying_ptr_uniform_unsigned
// CHECK: [[OFF:%.*]] = mul nuw i32 4, %{{.*}}
// CHECK: insertelement <8 x i32> {{.*}}, i32 [[OFF]], {{i32|i64}} 0
// CHECK: zext <8 x i32> {{%.*}} to <8 x i64>
// CHECK_WRAP-LABEL: define void @varying_ptr_uniform_unsigned
// CHECK_WRAP: mul i32 4, %{{.*}}
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

// p - u negates u as a signed value, so its offset keeps nsw and sext.
// CHECK-LABEL: define void @ptr_minus_unsigned
// CHECK: [[OFF:%.*]] = mul nsw <8 x i32> splat (i32 4), %negate_trunc
// CHECK: sext <8 x i32> [[OFF]] to <8 x i64>
void ptr_minus_unsigned(uniform float p[], uniform unsigned int32 idx[], uniform float r[]) {
    unsigned int32 u = idx[programIndex];
    uniform float *varying q = p + 1000 - u;
    r[programIndex] = *q;
}

// A 64-bit unsigned index is truncated with 32-bit addressing.
// CHECK-LABEL: define void @gather_unsigned64
// CHECK: [[IDX:%.*]] = trunc <8 x i64> %i_load to <8 x i32>
// CHECK: [[OFF:%.*]] = mul nuw <8 x i32> splat (i32 4), [[IDX]]
// CHECK: zext <8 x i32> [[OFF]] to <8 x i64>
void gather_unsigned64(uniform float p[], uniform unsigned int64 idx[], uniform float r[]) {
    unsigned int64 i = idx[programIndex];
    r[programIndex] = p[i];
}

// The soa block of an unsigned index must be computed with a logical shift:
// with an arithmetic one, an index of 2^31 or more gives a negative block
// number, and multiplying it with nuw gives poison.
struct Byte {
    uint8 x;
};
// CHECK-LABEL: @"soa_unsigned
// CHECK: [[MAJOR:%.*]] = lshr <8 x i32> {{%.*}}, splat (i32 4)
// CHECK: [[OFF:%.*]] = mul nuw <8 x i32> splat (i32 16), [[MAJOR]]
// CHECK: zext <8 x i32> [[OFF]] to <8 x i64>
uint8 soa_unsigned(soa<16> Byte *uniform p, unsigned int32 u) { return p[u].x; }

// CHECK-LABEL: @"soa_signed
// CHECK: [[MAJOR:%.*]] = ashr <8 x i32> {{%.*}}, splat (i32 4)
// CHECK: [[OFF:%.*]] = mul nsw <8 x i32> splat (i32 16), [[MAJOR]]
// CHECK: sext <8 x i32> [[OFF]] to <8 x i64>
uint8 soa_signed(soa<16> Byte *uniform p, int32 i) { return p[i].x; }

// A soa<> array indexed by an unsigned index takes the two-index path.
soa<16> Byte g_bytes[64];
// CHECK-LABEL: @soa_array_unsigned
// CHECK: [[MAJOR:%.*]] = lshr <8 x i32> {{%.*}}, splat (i32 4)
// CHECK: [[OFF:%.*]] = mul nuw <8 x i32> splat (i32 16), [[MAJOR]]
// CHECK: zext <8 x i32> [[OFF]] to <8 x i64>
uint8 soa_array_unsigned(unsigned int32 u) { return g_bytes[u].x; }
