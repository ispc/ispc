// This test checks for the use of the no signed wrap (`nsw`) attribute for arithmetic
// expressions which may suffer from overflow.
// Needed for induction variable simplification dicussed in
// [GitHub issue ISPC/#2460](https://github.com/ispc/ispc/issues/2460).

// RUN: %{ispc} %s --nostdlib --emit-llvm-text --target=host -o - |  FileCheck %s --check-prefix=CHECK_NSW

// REQUIRES: !XE_ENABLED

// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @mul_64
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @mul_32
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @mul_16
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @mul_8
// CHECK_NSW-COUNT-1:         %{{.*}}= mul{{.*}}nsw{{.*}}i8

unmasked int64 mul_64(int64 a, int64 b) { return a * b; }

unmasked int32 mul_32(int32 a, int32 b) { return a * b; }

unmasked int16 mul_16(int16 a, int16 b) { return a * b; }

unmasked int8 mul_8(int8 a, int8 b) { return a * b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @mul_u64
// CHECK_NSW-NOT:             %{{.*}}= mul{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @mul_u32
// CHECK_NSW-NOT:             %{{.*}}= mul{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @mul_u16
// CHECK_NSW-NOT:             %{{.*}}= mul{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @mul_u8
// CHECK_NSW-NOT:             %{{.*}}= mul{{.*}}nsw{{.*}}i8

unmasked uint64 mul_u64(uint64 a, uint64 b) { return a * b; }

unmasked uint32 mul_u32(uint32 a, uint32 b) { return a * b; }

unmasked uint16 mul_u16(uint16 a, uint16 b) { return a * b; }

unmasked uint8 mul_u8(uint8 a, uint8 b) { return a * b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @add_64
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @add_32
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @add_16
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @add_8
// CHECK_NSW-COUNT-1:         %{{.*}}= add{{.*}}nsw{{.*}}i8

unmasked int64 add_64(int64 a, int64 b) { return a + b; }

unmasked int32 add_32(int32 a, int32 b) { return a + b; }

unmasked int16 add_16(int16 a, int16 b) { return a + b; }

unmasked int8 add_8(int8 a, int8 b) { return a + b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @add_u64
// CHECK_NSW-NOT:             %{{.*}}= add{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @add_u32
// CHECK_NSW-NOT:             %{{.*}}= add{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @add_u16
// CHECK_NSW-NOT:             %{{.*}}= add{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @add_u8
// CHECK_NSW-NOT:             %{{.*}}= add{{.*}}nsw{{.*}}i8

unmasked uint64 add_u64(uint64 a, uint64 b) { return a + b; }

unmasked uint32 add_u32(uint32 a, uint32 b) { return a + b; }

unmasked uint16 add_u16(uint16 a, uint16 b) { return a + b; }

unmasked uint8 add_u8(uint8 a, uint8 b) { return a + b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @sub_64
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @sub_32
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @sub_16
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @sub_8
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i8

unmasked int64 sub_64(int64 a, int64 b) { return a - b; }

unmasked int32 sub_32(int32 a, int32 b) { return a - b; }

unmasked int16 sub_16(int16 a, int16 b) { return a - b; }

unmasked int8 sub_8(int8 a, int8 b) { return a - b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @sub_u64
// CHECK_NSW-NOT:             %{{.*}}= sub{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @sub_u32
// CHECK_NSW-NOT:             %{{.*}}= sub{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @sub_u16
// CHECK_NSW-NOT:             %{{.*}}= sub{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @sub_u8
// CHECK_NSW-NOT:             %{{.*}}= sub{{.*}}nsw{{.*}}i8

unmasked uint64 sub_u64(uint64 a, uint64 b) { return a - b; }

unmasked uint32 sub_u32(uint32 a, uint32 b) { return a - b; }

unmasked uint16 sub_u16(uint16 a, uint16 b) { return a - b; }

unmasked uint8 sub_u8(uint8 a, uint8 b) { return a - b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @shl_64
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @shl_32
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @shl_16
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @shl_8
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i8

unmasked int64 shl_64(int64 a, int64 b) { return a << b; }

unmasked int32 shl_32(int32 a, int32 b) { return a << b; }

unmasked int16 shl_16(int16 a, int16 b) { return a << b; }

unmasked int8 shl_8(int8 a, int8 b) { return a << b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @shl_u64
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @shl_u32
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @shl_u16
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @shl_u8
// CHECK_NSW-NOT:             %{{.*}}= shl{{.*}}nsw{{.*}}i8

unmasked uint64 shl_u64(uint64 a, uint64 b) { return a << b; }

unmasked uint32 shl_u32(uint32 a, uint32 b) { return a << b; }

unmasked uint16 shl_u16(uint16 a, uint16 b) { return a << b; }

unmasked uint8 shl_u8(uint8 a, uint8 b) { return a << b; }


// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i64> @neg_64
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i64
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i32> @neg_32
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i32
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i16> @neg_16
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i16
// CHECK_NSW-LABEL:         define <{{[0-9]*}} x i8> @neg_8
// CHECK_NSW-COUNT-1:         %{{.*}}= sub{{.*}}nsw{{.*}}i8

unmasked int64 neg_64(int64 a) { return -a; }

unmasked int32 neg_32(int32 a) { return -a; }

unmasked int16 neg_16(int16 a) { return -a; }

unmasked int8 neg_8(int8 a) { return -a; }


// CHECK_NSW-LABEL:         define {{.*}} @ptrMath_64
// CHECK_NSW-NOT:             nsw
// CHECK_NSW-LABEL:         define {{.*}} @ptrMath_32
// CHECK_NSW-NOT:             nsw
// CHECK_NSW-LABEL:         define {{.*}} @ptrMath_16
// CHECK_NSW-NOT:             nsw
// CHECK_NSW-LABEL:         define {{.*}} @ptrMath_8
// CHECK_NSW-NOT:             nsw

unmasked uniform int64 * uniform ptrMath_64(uniform int64 * uniform a, uniform int64 k) { return &a[0] + k; }

unmasked uniform int32 * uniform ptrMath_32(uniform int32 * uniform a, uniform int32 k) { return &a[0] + k; }

unmasked uniform int16 * uniform ptrMath_16(uniform int16 * uniform a, uniform int16 k) { return &a[0] + k; }

unmasked uniform int8 * uniform ptrMath_8(uniform int8 * uniform a, uniform int8 k) { return &a[0] + k; }
