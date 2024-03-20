// Test checks emitted code for VNNI dot product instructions.
// RUN: %{ispc} %s --target=avx1-i64x4 --cpu=adl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM
// RUN: %{ispc} %s --target=avx2-i32x4 --cpu=adl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM
// RUN: %{ispc} %s --target=avx2-i64x4 --cpu=adl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM

// RUN: %{ispc} %s --target=avx1-i32x16 --cpu=adl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMMX2
// RUN: %{ispc} %s --target=avx2-i32x16 --cpu=adl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMMX2
// RUN: %{ispc} %s --target=avx2-i16x16 --cpu=adl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMMX2

// RUN: %{ispc} %s --target=avx2-i8x32 --cpu=adl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMMX4

// RUN: %{ispc} %s --target=avx512skx-x4 --cpu=icl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM
// RUN: %{ispc} %s --target=avx512skx-x4 --cpu=icx --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM
// RUN: %{ispc} %s --target=avx512skx-x4 --cpu=tgl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM
// RUN: %{ispc} %s --target=avx512spr-x4 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM

// RUN: %{ispc} %s --target=avx512skx-x8 --cpu=icl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMM
// RUN: %{ispc} %s --target=avx512skx-x8 --cpu=icx --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMM
// RUN: %{ispc} %s --target=avx512skx-x8 --cpu=tgl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMM
// RUN: %{ispc} %s --target=avx512spr-x8 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMM

// RUN: %{ispc} %s --target=avx512skx-x16 --cpu=icl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMM
// RUN: %{ispc} %s --target=avx512skx-x16 --cpu=icx --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMM
// RUN: %{ispc} %s --target=avx512skx-x16 --cpu=tgl --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMM
// RUN: %{ispc} %s --target=avx512spr-x16 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMM

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK_ALL-LABEL: dot4add_u8i8
// CHECK_XMM: vpdpbusd	{{.*}} %xmm
// CHECK_YMM: vpdpbusd	{{.*}} %ymm
// CHECK_YMMX2-COUNT-2: vpdpbusd {{.*}} %ymm
// CHECK_YMMX4-COUNT-4: vpdpbusd {{.*}} %ymm
// CHECK_ZMM: vpdpbusd	{{.*}} %zmm
void dot4add_u8i8(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot4add_u8i8_sat
// CHECK_XMM: vpdpbusds	{{.*}} %xmm
// CHECK_YMM: vpdpbusds	{{.*}} %ymm
// CHECK_YMMX2-COUNT-2: vpdpbusds {{.*}} %ymm
// CHECK_YMMX4-COUNT-4: vpdpbusds {{.*}} %ymm
// CHECK_ZMM: vpdpbusds	{{.*}} %zmm
void dot4add_u8i8_sat(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed_sat(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot2add_i16
// CHECK_XMM: vpdpwssd	{{.*}} %xmm
// CHECK_YMM: vpdpwssd	{{.*}} %ymm
// CHECK_YMMX2-COUNT-2: vpdpwssd {{.*}} %ymm
// CHECK_YMMX4-COUNT-4: vpdpwssd {{.*}} %ymm
// CHECK_ZMM: vpdpwssd	{{.*}} %zmm
void dot2add_i16(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot2add_i16packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot2add_i16_sat
// CHECK_XMM: vpdpwssds	{{.*}} %xmm
// CHECK_YMM: vpdpwssds	{{.*}} %ymm
// CHECK_YMMX2-COUNT-2: vpdpwssds {{.*}} %ymm
// CHECK_YMMX4-COUNT-4: vpdpwssds {{.*}} %ymm
// CHECK_ZMM: vpdpwssds	{{.*}} %zmm
void dot2add_i16_sat(uniform int a[], uniform int b[], uniform int dst[]) {
    dst[programIndex] = dot2add_i16packed_sat(a[programIndex], b[programIndex], programIndex);
}
