// Test shuffle asm codegen inside foreach loop with reads/writes and computation
// Key Shuffle Instructions Used:
// - i8: vpshufb for smaller vectors, transitions to vpermw/vpermi2w for larger vectors
// - i16/half: vpermw consistently, vpermi2w for x64
// - i32/float: vpermilps → vpermd → vpermi2d → vpgatherdd as vector width increases
// - i64/double: vpermq → vpermt2q → vpermt2pd → vpgatherdq as vector width increases
//
// Anti-patterns Checked:
// - Ensures no old inefficient vpextrd/vpinsrb/vpinsrw instructions are generated

// RUN: %{ispc} %s --target=avx512skx-x4 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X4
// RUN: %{ispc} %s --target=avx512skx-x8 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X8
// RUN: %{ispc} %s --target=avx512skx-x16 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X16
// RUN: %{ispc} %s --target=avx512skx-x32 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X32
// RUN: %{ispc} %s --target=avx512skx-x64 --emit-asm -O2 -o - | FileCheck %s --check-prefix=CHECK-X64

// REQUIRES: X86_ENABLED

#define PERMUTE_VAL 1

// CHECK-X4-LABEL: shuffle_i8___un_3C_unt_3E_un_3C_unt_3E_vyiuni:
// CHECK-X4-DAG: vpshufb
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrb
// CHECK-X8-LABEL: shuffle_i8___un_3C_unt_3E_un_3C_unt_3E_vyiuni:
// CHECK-X8-DAG: vpshufb
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrb
// CHECK-X16-LABEL: shuffle_i8___un_3C_unt_3E_un_3C_unt_3E_vyiuni:
// CHECK-X16-DAG: vpshufb
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrb
// CHECK-X32-LABEL: shuffle_i8___un_3C_unt_3E_un_3C_unt_3E_vyiuni:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrb
// CHECK-X64-LABEL: shuffle_i8___un_3C_unt_3E_un_3C_unt_3E_vyiuni:
// CHECK-X64-DAG: vpermt2w
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrb
void shuffle_i8(uniform int8 a[], uniform int8 ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}

// CHECK-X4-LABEL: shuffle_i16___un_3C_uns_3E_un_3C_uns_3E_vyiuni:
// CHECK-X4-DAG: vpermw
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrw
// CHECK-X8-LABEL: shuffle_i16___un_3C_uns_3E_un_3C_uns_3E_vyiuni:
// CHECK-X8-DAG: vpermw
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrw
// CHECK-X16-LABEL: shuffle_i16___un_3C_uns_3E_un_3C_uns_3E_vyiuni:
// CHECK-X16-DAG: vpermw
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrw
// CHECK-X32-LABEL: shuffle_i16___un_3C_uns_3E_un_3C_uns_3E_vyiuni:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrw
// CHECK-X64-LABEL: shuffle_i16___un_3C_uns_3E_un_3C_uns_3E_vyiuni:
// CHECK-X64-DAG: vpermt2w
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrw
void shuffle_i16(uniform int16 a[], uniform int16 ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}

// CHECK-X4-LABEL: shuffle_i32___un_3C_uni_3E_un_3C_uni_3E_vyiuni:
// CHECK-X4-DAG: vpermilps
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrd
// CHECK-X8-LABEL: shuffle_i32___un_3C_uni_3E_un_3C_uni_3E_vyiuni:
// CHECK-X8-DAG: vpermd
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrd
// CHECK-X16-LABEL: shuffle_i32___un_3C_uni_3E_un_3C_uni_3E_vyiuni:
// CHECK-X16-DAG: vpermd
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrd
// CHECK-X32-LABEL: shuffle_i32___un_3C_uni_3E_un_3C_uni_3E_vyiuni:
// CHECK-X32-DAG: vpermt2d
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrd
// CHECK-X64-LABEL: shuffle_i32___un_3C_uni_3E_un_3C_uni_3E_vyiuni:
// CHECK-X64-DAG: vpgatherqd
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrd
void shuffle_i32(uniform int32 a[], uniform int32 ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}

// CHECK-X4-LABEL: shuffle_i64___un_3C_unI_3E_un_3C_unI_3E_vyiuni:
// CHECK-X4-DAG: vpermq
// CHECK-X4-NOT: vpextrd
// CHECK-X8-LABEL: shuffle_i64___un_3C_unI_3E_un_3C_unI_3E_vyiuni:
// CHECK-X8-DAG: vpermt2q
// CHECK-X8-NOT: vpextrd
// CHECK-X16-LABEL: shuffle_i64___un_3C_unI_3E_un_3C_unI_3E_vyiuni:
// CHECK-X16-DAG: vpermt2q
// CHECK-X16-NOT: vpextrd
// CHECK-X32-LABEL: shuffle_i64___un_3C_unI_3E_un_3C_unI_3E_vyiuni:
// CHECK-X32-DAG: vpgatherqq
// CHECK-X32-NOT: vpextrd
// CHECK-X64-LABEL: shuffle_i64___un_3C_unI_3E_un_3C_unI_3E_vyiuni:
// CHECK-X64-DAG: vpgatherqq
// CHECK-X64-NOT: vpextrd
void shuffle_i64(uniform int64 a[], uniform int64 ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}

// CHECK-X4-LABEL: shuffle_float16___un_3C_unh_3E_un_3C_unh_3E_vyiuni:
// CHECK-X4-DAG: vpermw
// CHECK-X4-NOT: vpextrd
// CHECK-X4-NOT: vpinsrw
// CHECK-X8-LABEL: shuffle_float16___un_3C_unh_3E_un_3C_unh_3E_vyiuni:
// CHECK-X8-DAG: vpermw
// CHECK-X8-NOT: vpextrd
// CHECK-X8-NOT: vpinsrw
// CHECK-X16-LABEL: shuffle_float16___un_3C_unh_3E_un_3C_unh_3E_vyiuni:
// CHECK-X16-DAG: vpermw
// CHECK-X16-NOT: vpextrd
// CHECK-X16-NOT: vpinsrw
// CHECK-X32-LABEL: shuffle_float16___un_3C_unh_3E_un_3C_unh_3E_vyiuni:
// CHECK-X32-DAG: vpermw
// CHECK-X32-NOT: vpextrd
// CHECK-X32-NOT: vpinsrw
// CHECK-X64-LABEL: shuffle_float16___un_3C_unh_3E_un_3C_unh_3E_vyiuni:
// CHECK-X64-DAG: vpermt2w
// CHECK-X64-NOT: vpextrd
// CHECK-X64-NOT: vpinsrw
void shuffle_float16(uniform float16 a[], uniform float16 ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}

// CHECK-X4-LABEL: shuffle_float___un_3C_unf_3E_un_3C_unf_3E_vyiuni:
// CHECK-X4-DAG: vpermilps
// CHECK-X4-NOT: vpextrd
// CHECK-X8-LABEL: shuffle_float___un_3C_unf_3E_un_3C_unf_3E_vyiuni:
// CHECK-X8-DAG: vpermd
// CHECK-X8-NOT: vpextrd
// CHECK-X16-LABEL: shuffle_float___un_3C_unf_3E_un_3C_unf_3E_vyiuni:
// CHECK-X16-DAG: vpermd
// CHECK-X16-NOT: vpextrd
// CHECK-X32-LABEL: shuffle_float___un_3C_unf_3E_un_3C_unf_3E_vyiuni:
// CHECK-X32-DAG: vpermt2ps
// CHECK-X32-NOT: vpextrd
// CHECK-X64-LABEL: shuffle_float___un_3C_unf_3E_un_3C_unf_3E_vyiuni:
// CHECK-X64-DAG: vpgatherqd
// CHECK-X64-NOT: vpextrd
void shuffle_float(uniform float a[], uniform float ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}

// CHECK-X4-LABEL: shuffle_double___un_3C_und_3E_un_3C_und_3E_vyiuni:
// CHECK-X4-DAG: vpermq
// CHECK-X4-NOT: vpextrd
// CHECK-X8-LABEL: shuffle_double___un_3C_und_3E_un_3C_und_3E_vyiuni:
// CHECK-X8-DAG: vpermt2pd
// CHECK-X8-NOT: vpextrd
// CHECK-X16-LABEL: shuffle_double___un_3C_und_3E_un_3C_und_3E_vyiuni:
// CHECK-X16-DAG: vpermt2pd
// CHECK-X16-NOT: vpextrd
// CHECK-X32-LABEL: shuffle_double___un_3C_und_3E_un_3C_und_3E_vyiuni:
// CHECK-X32-DAG: vpgatherqq
// CHECK-X32-NOT: vpextrd
// CHECK-X64-LABEL: shuffle_double___un_3C_und_3E_un_3C_und_3E_vyiuni:
// CHECK-X64-DAG: vpgatherqq
// CHECK-X64-NOT: vpextrd
void shuffle_double(uniform double a[], uniform double ret[], int perm, uniform int count) {
    assume(count % 64 == 0);
    foreach (i = 0 ... count) {
        ret[i] = shuffle(a[i], programIndex + perm - PERMUTE_VAL);
    }
}
