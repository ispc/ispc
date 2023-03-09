// RUN: %{ispc} %s --nostdlib --nowrap --target=sse2-i32x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse2-i32x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4-i8x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4-i16x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4-i32x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4-i32x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE

// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4.1-i8x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4.1-i16x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4.1-i32x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=sse4.1-i32x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE

// RUN: %{ispc} %s --nostdlib --nowrap --target=avx1-i32x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx1-i32x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx1-i32x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx1-i64x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_SSE

// RUN: %{ispc} %s --nostdlib --nowrap --target=avx2-i8x32 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx2-i16x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx2-i32x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx2-i32x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx2-i32x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx2-i64x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX2

// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512knl-x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512skx-x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512skx-x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512skx-x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512skx-x32 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512skx-x64 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512spr-x4 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512spr-x8 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512spr-x16 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512spr-x32 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512
// RUN: %{ispc} %s --nostdlib --nowrap --target=avx512spr-x64 -o %t.o 2>&1 | FileCheck %s -check-prefix=CHECK_AVX512

// A note about notrunning on macOS - it's because of SPR not being supported there.
// It should be sufficient to run this test on other platforms and not to complicate
// with splitting it to SPR and non-SPR versions.
// REQUIRES: X86_ENABLED && LLVM_14_0+ && !MACOS_HOST

// Float16 -> integers

int8 cvt_fp16_to_i8(float16 f) { return f; }

uint8 cvt_fp16_to_ui8(float16 f) { return f; }

int16 cvt_fp16_to_i16(float16 f) { return f; }

uint16 cvt_fp16_to_ui16(float16 f) { return f; }

int32 cvt_fp16_to_i32(float16 f) { return f; }

// CHECK_AVX2: Performance Warning: Conversion from float16 to uint32 is slow. Use "int32" if possible
uint32 cvt_fp16_to_ui32(float16 f) { return f; }

int64 cvt_fp16_to_i64(float16 f) { return f; }

// CHECK_AVX2: Performance Warning: Conversion from float16 to uint64 is slow. Use "int64" if possible
uint64 cvt_fp16_to_ui64(float16 f) { return f; }

// Float -> integers

int8 cvt_fp32_to_i8(float f) { return f; }

uint8 cvt_fp32_to_ui8(float f) { return f; }

int16 cvt_fp32_to_i16(float f) { return f; }

uint16 cvt_fp32_to_ui16(float f) { return f; }

int32 cvt_fp32_to_i32(float f) { return f; }

// CHECK_SSE: Performance Warning: Conversion from float to uint32 is slow. Use "int32" if possible
// CHECK_AVX2: Performance Warning: Conversion from float to uint32 is slow. Use "int32" if possible
uint32 cvt_fp32_to_ui32(float f) { return f; }

int64 cvt_fp32_to_i64(float f) { return f; }

// CHECK_SSE: Performance Warning: Conversion from float to uint64 is slow. Use "int64" if possible
// CHECK_AVX2: Performance Warning: Conversion from float to uint64 is slow. Use "int64" if possible
uint64 cvt_fp32_to_ui64(float f) { return f; }

// Double -> integers

int8 cvt_fp64_to_i8(double f) { return f; }

uint8 cvt_fp64_to_ui8(double f) { return f; }

int16 cvt_fp64_to_i16(double f) { return f; }

uint16 cvt_fp64_to_ui16(double f) { return f; }

int32 cvt_fp64_to_i32(double f) { return f; }

// CHECK_SSE: Performance Warning: Conversion from double to uint32 is slow. Use "int32" if possible
// CHECK_AVX2 Performance Warning: Conversion from double to uint32 is slow. Use "int32" if possible
uint32 cvt_fp64_to_ui32(double f) { return f; }

int64 cvt_fp64_to_i64(double f) { return f; }

// CHECK_SSE: Performance Warning: Conversion from double to uint64 is slow. Use "int64" if possible
// CHECK_AVX2: Performance Warning: Conversion from double to uint64 is slow. Use "int64" if possible
uint64 cvt_fp64_to_ui64(double f) { return f; }

// Integers -> float16

float16 cvt_i8_to_fp16(int8 i) { return i; }

float16 cvt_ui8_to_fp16(uint8 i) { return i; }

float16 cvt_i16_to_fp16(int16 i) { return i; }

float16 cvt_ui16_to_fp16(uint16 i) { return i; }

float16 cvt_i32_to_fp16(int32 i) { return i; }
// CHECK_AVX2: Performance Warning: Conversion from uint32 to float16 is slow. Use "int32" if possible
float16 cvt_ui32_to_fp16(uint32 i) { return i; }

float16 cvt_i64_to_fp16(int64 i) { return i; }
// CHECK_AVX2: Performance Warning: Conversion from uint64 to float16 is slow. Use "int32" if possible
float16 cvt_ui64_to_fp16(uint64 i) { return i; }

// Integers -> float

float cvt_i8_to_fp32(int8 i) { return i; }

float cvt_ui8_to_fp32(uint8 i) { return i; }

float cvt_i16_to_fp32(int16 i) { return i; }

float cvt_ui16_to_fp32(uint16 i) { return i; }

float cvt_i32_to_fp32(int32 i) { return i; }

// CHECK_SSE: Performance Warning: Conversion from uint32 to float is slow. Use "int32" if possible
// CHECK_AVX2: Performance Warning: Conversion from uint32 to float is slow. Use "int32" if possible
float cvt_ui32_to_fp32(uint32 i) { return i; }

float cvt_i64_to_fp32(int64 i) { return i; }

// CHECK_SSE: Performance Warning: Conversion from uint64 to float is slow. Use "int64" if possible
// CHECK_AVX2: Performance Warning: Conversion from uint64 to float is slow. Use "int64" if possible
float cvt_ui64_to_fp32(uint64 i) { return i; }

// Integers -> double

double cvt_i8_to_fp64(int8 i) { return i; }

double cvt_ui8_to_fp64(uint8 i) { return i; }

double cvt_i16_to_fp64(int16 i) { return i; }

double cvt_ui16_to_fp64(uint16 i) { return i; }

double cvt_i32_to_fp64(int32 i) { return i; }

// CHECK_SSE: Performance Warning: Conversion from uint32 to double is slow. Use "int32" if possible
// CHECK_AVX2: Performance Warning: Conversion from uint32 to double is slow. Use "int32" if possible
double cvt_ui32_to_fp64(uint32 i) { return i; }

double cvt_i64_to_fp64(int64 i) { return i; }

// CHECK_SSE: Performance Warning: Conversion from uint64 to double is slow. Use "int32" if possible
// CHECK_AVX2: Performance Warning: Conversion from uint64 to double is slow. Use "int32" if possible
double cvt_ui64_to_fp64(uint64 i) { return i; }

// Integer div/mod, no need to check all types.

// CHECK_SSE-COUNT-2: Performance Warning: Division with varying integer types is very inefficient.
// CHECK_AVX2-COUNT-2: Performance Warning: Division with varying integer types is very inefficient.
// CHECK_AVX512-COUNT-2: Performance Warning: Division with varying integer types is very inefficient.
int32 div_i32(int32 a, int32 b) { return a / b; }
uint32 div_ui32(uint32 a, uint32 b) { return a / b; }

// CHECK_SSE-COUNT-2: Performance Warning: Modulus operator with varying types is very inefficient.
// CHECK_AVX2-COUNT-2: Performance Warning: Modulus operator with varying types is very inefficient.
// CHECK_AVX512-COUNT-2: Performance Warning: Modulus operator with varying types is very inefficient.
int32 mod_i32(int32 a, int32 b) { return a % b; }
uint32 mod_ui32(uint32 a, uint32 b) { return a % b; }

// Shift Right by variable amount

// CHECK_SSE: Performance Warning: Shift right is inefficient for varying shift amounts.
int32 shr_i32(int32 a, int32 b) { return a >> b; }
