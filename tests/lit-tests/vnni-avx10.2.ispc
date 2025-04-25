// Test checks emitted code for VNNI dot product instructions.

// RUN: %{ispc} %s --target=avx10.2-x4 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_XMM
// RUN: %{ispc} %s --target=avx10.2-x8 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_YMM
// RUN: %{ispc} %s --target=avx10.2-x16 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMM
// RUN: %{ispc} %s --target=avx10.2-x32 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMMX2
// RUN: %{ispc} %s --target=avx10.2-x64 --emit-asm -o - | FileCheck %s -check-prefixes=CHECK_ALL,CHECK_ZMMX4

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+

// CHECK_ALL-LABEL: dot4add_i8i8
// CHECK_XMM: vpdpbssd	{{.*}} %xmm
// CHECK_YMM: vpdpbssd	{{.*}} %ymm
// CHECK_ZMM: vpdpbssd	{{.*}} %zmm
// CEHCK_ZMMX2-COUNT-2: vpdpbssd {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpbssd {{.*}} %zmm
unmasked void dot4add_i8i8(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_i8i8packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot4add_i8i8_sat
// CHECK_XMM: vpdpbssds	{{.*}} %xmm
// CHECK_YMM: vpdpbssds	{{.*}} %ymm
// CHECK_ZMM: vpdpbssds	{{.*}} %zmm
// CHECK_ZMMX2-COUNT-2: vpdpbssds {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpbssds {{.*}} %zmm
unmasked void dot4add_i8i8_sat(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_i8i8packed_sat(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot4add_u8u8
// CHECK_XMM: vpdpbuud	{{.*}} %xmm
// CHECK_YMM: vpdpbuud	{{.*}} %ymm
// CHECK_ZMM: vpdpbuud	{{.*}} %zmm
// CHECK_ZMMX2-COUNT-2: vpdpbuud {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpbuud {{.*}} %zmm
unmasked void dot4add_u8u8(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8u8packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot4add_u8u8_sat
// CHECK_XMM: vpdpbuuds	{{.*}} %xmm
// CHECK_YMM: vpdpbuuds	{{.*}} %ymm
// CHECK_ZMM: vpdpbuuds	{{.*}} %zmm
// CHECK_ZMMX2-COUNT-2: vpdpbuuds {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpbuuds {{.*}} %zmm
unmasked void dot4add_u8u8_sat(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8u8packed_sat(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot2add_u16u16
// CHECK_XMM: vpdpwuud	{{.*}} %xmm
// CHECK_YMM: vpdpwuud	{{.*}} %ymm
// CHECK_ZMM: vpdpwuud	{{.*}} %zmm
// CHECK_ZMMX2-COUNT-2: vpdpwuud {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpwuud {{.*}} %zmm
unmasked void dot2add_u16u16(uniform uint a[], uniform uint b[], uniform uint dst[]) {
    dst[programIndex] = dot2add_u16u16packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot2add_u16u16_sat
// CHECK_XMM: vpdpwuuds	{{.*}} %xmm
// CHECK_YMM: vpdpwuuds	{{.*}} %ymm
// CHECK_ZMM: vpdpwuuds	{{.*}} %zmm
// CHECK_ZMMX2-COUNT-2: vpdpwuuds {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpwuuds {{.*}} %zmm
unmasked void dot2add_u16u16_sat(uniform uint a[], uniform uint b[], uniform uint dst[]) {
    dst[programIndex] = dot2add_u16u16packed_sat(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot2add_u16i16
// CHECK_XMM: vpdpwusd	{{.*}} %xmm
// CHECK_YMM: vpdpwusd	{{.*}} %ymm
// CHECK_ZMM: vpdpwusd	{{.*}} %zmm
// CHECK_ZMMX2-COUNT-2: vpdpwusd {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpwusd {{.*}} %zmm
unmasked void dot2add_u16i16(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot2add_u16i16packed(a[programIndex], b[programIndex], programIndex);
}

// CHECK_ALL-LABEL: dot2add_u16i16_sat
// CHECK_XMM: vpdpwusds	{{.*}} %xmm
// CHECK_YMM: vpdpwusds	{{.*}} %ymm
// CHECK_ZMM: vpdpwusds	{{.*}} %zmm
// CHECK_ZMMX2-COUNT-2: vpdpwusds {{.*}} %zmm
// CHECK_ZMMX4-COUNT-4: vpdpwusds {{.*}} %zmm
unmasked void dot2add_u16i16_sat(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot2add_u16i16packed_sat(a[programIndex], b[programIndex], programIndex);
}