// Test checks emitted code for VNNI dot product instructions.

// RUN: %{ispc} %s --target=avx512icl-x32 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMMX2
// RUN: %{ispc} %s --target=avx512spr-x32 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMMX2

// RUN: %{ispc} %s --target=avx512icl-x64 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMMX4
// RUN: %{ispc} %s --target=avx512spr-x64 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMMX4

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK_ALL-LABEL: dot4add_u8i8
// CHECK_ZMMX2-COUNT-2: vpdpbusd {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpbusd {{.*}} %zmm
void dot4add_u8i8(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot4add_u8i8_sat
// CHECK_ZMMX2-COUNT-2: vpdpbusds {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpbusds {{.*}} %zmm
void dot4add_u8i8_sat(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed_sat(a[programIndex], b[programIndex], programIndex);
}

// For this test "unmasked" version of function is used. This is because for avx512*x32 and avx512*x64 targets
// the generic masked stores are used resulting in a set of per-lane scalar stores. In this case it's more profitable
// to transform vpdpwssd into vpmaddwd+vpaddd https://reviews.llvm.org/D148980.
// Related discussion: https://github.com/llvm/llvm-project/issues/84182

// CHECK_ALL-LABEL: dot2add_i16
// CHECK_ZMMX2-COUNT-2: vpdpwssd {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpwssd {{.*}} %zmm
unmasked void dot2add_i16(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot2add_i16packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot2add_i16_sat
// CHECK_ZMMX2-COUNT-2: vpdpwssds {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpwssds {{.*}} %zmm
void dot2add_i16_sat(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot2add_i16packed_sat(a[programIndex], b[programIndex], programIndex);
}
