// The test is checking that vperm<type> instruction is generated for shuffle(T, int) on avx2 and avx512 targets.

// RUN: %{ispc} --target=avx2-i32x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx2vnni-i32x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT32,CHECK_FLOAT

// RUN: %{ispc} --target=avx512icl-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT32,CHECK_FLOAT,CHECK_INT64,CHECK_DOUBLE
// RUN: %{ispc} --target=avx512skx-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT32,CHECK_FLOAT,CHECK_INT64,CHECK_DOUBLE
// RUN: %{ispc} --target=avx512spr-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT32,CHECK_FLOAT,CHECK_INT64,CHECK_DOUBLE

// RUN: %{ispc} --target=avx512icl-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512skx-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512spr-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT16,CHECK_INT32,CHECK_FLOAT

// RUN: %{ispc} --target=avx512icl-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512skx-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512spr-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT16,CHECK_INT32,CHECK_FLOAT

// RUN: %{ispc} --target=avx512icl-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512skx-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512spr-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT


// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK_ALL-LABEL: Test_int8
// CHECK_INT8: vperm
unmasked int8 Test_int8(uniform int8 *uniform data, int32 j) {
    return shuffle(data[programIndex + programCount], j);
}

// CHECK_ALL-LABEL: Test_int16
// CHECK_INT16: vperm
unmasked int16 Test_int16(uniform int16 *uniform data, int32 j) {
    return shuffle(data[programIndex + programCount], j);
}

// CHECK_ALL-LABEL: Test_int32
// CHECK_INT32: vperm
unmasked int32 Test_int32(uniform int32 *uniform data, int32 j) {
    return shuffle(data[programIndex + programCount], j);
}

// CHECK_ALL-LABEL: Test_float
// CHECK_FLOAT: vperm
unmasked float Test_float(uniform float *uniform data, int32 j) {
    return shuffle(data[programIndex + programCount], j);
}

// CHECK_ALL-LABEL: Test_double
// CHECK_DOUBLE: vperm
unmasked double Test_double(uniform double *uniform data, int32 j) {
    return shuffle(data[programIndex + programCount], j);
}

// CHECK_ALL-LABEL: Test_int64
// CHECK_INT64: vperm
unmasked int64 Test_int64(uniform int64 *uniform data, int32 j) {
    return shuffle(data[programIndex + programCount], j);
}