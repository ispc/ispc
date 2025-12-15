// The test is checking that vpshufb or vperm<type> instruction is generated for shuffle(T, T, int) on avx512 targets.

// RUN: %{ispc} --target=avx512skx-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT,CHECK_DOUBLE,CHECK_INT64
// RUN: %{ispc} --target=avx512icl-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT,CHECK_DOUBLE,CHECK_INT64
// RUN: %{ispc} --target=avx512spr-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT,CHECK_DOUBLE,CHECK_INT64
// RUN: %{ispc} --target=avx512gnr-x4 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT,CHECK_DOUBLE,CHECK_INT64
// RUN: %{ispc} --target=avx512skx-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512icl-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512spr-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512gnr-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o -  | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_SKX_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512skx-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512icl-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512spr-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512gnr-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8,CHECK_INT16,CHECK_INT32,CHECK_FLOAT
// RUN: %{ispc} --target=avx512skx-x32 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8
// RUN: %{ispc} --target=avx512icl-x32 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8
// RUN: %{ispc} --target=avx512spr-x32 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8
// RUN: %{ispc} --target=avx512gnr-x32 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8
// RUN: %{ispc} --target=avx512icl-x64 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8
// RUN: %{ispc} --target=avx512spr-x64 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8
// RUN: %{ispc} --target=avx512gnr-x64 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_INT8

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK_ALL-LABEL: Shuffle_i8___
// CHECK_INT8: vperm
// CHECK_SKX_INT8: vpshufb
unmasked varying int8 Shuffle_i8(varying int8 a, varying int8 b, varying int32 perm) {
    return shuffle(a, b, perm);
}

// CHECK_ALL-LABEL: Shuffle_i16___
// CHECK_INT16: vperm
unmasked varying int16 Shuffle_i16(varying int16 a, varying int16 b, varying int32 perm) {
    return shuffle(a, b, perm);
}

// CHECK_ALL-LABEL: Shuffle_i32___
// CHECK_INT32: vperm
unmasked varying int32 Shuffle_i32(varying int32 a, varying int32 b, varying int32 perm) {
    return shuffle(a, b, perm);
}

// CHECK_ALL-LABEL: Shuffle_float___
// CHECK_FLOAT: vperm
unmasked varying float Shuffle_float(varying float a, varying float b, varying int32 perm) {
    return shuffle(a, b, perm);
}

// CHECK_ALL-LABEL: Shuffle_double___
// CHECK_DOUBLE: vperm
unmasked varying double Shuffle_double(varying double a, varying double b, varying int32 perm) {
    return shuffle(a, b, perm);
}

// CHECK_ALL-LABEL: Shuffle_i64___
// CHECK_INT64: vperm
unmasked varying int64 Shuffle_i64(varying int64 a, varying int64 b, varying int32 perm) {
    return shuffle(a, b, perm);
}
