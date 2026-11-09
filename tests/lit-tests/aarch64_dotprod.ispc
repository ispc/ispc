// Test checks emitted code for ARM (aarch64) dot product instructions.

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --cpu=cortex-a55 --emit-asm  -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_DOT_ONLY,DOT_X4
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --cpu=cortex-a510 --emit-asm  -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_MIXED_SIGN

// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --cpu=cortex-a55 --emit-asm  -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_DOT_ONLY,DOT_X8
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --cpu=cortex-a510 --emit-asm  -o - | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_MIXED_SIGN

// The 8-bit and 16-bit mask targets take the same paths with more vectors per gang: neon-i16x8 has two vectors
// of four 32-bit lanes and neon-i8x16 has four.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i16x8 --cpu=cortex-a55 --emit-asm  -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_DOT_ONLY,DOT_X8
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i8x16 --cpu=cortex-a55 --emit-asm  -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_DOT_ONLY,DOT_X16
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i8x16 --cpu=cortex-a510 --emit-asm  -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK_ALL,CHECK_MIXED_SIGN

// Test that on HW not supporting dot product natively *dot instructions are not produced.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --cpu=cortex-a35 --emit-asm  -o - | FileCheck %s --check-prefix=CHECK_NO_HW_DOT
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --cpu=cortex-a53 --emit-asm  -o - | FileCheck %s --check-prefix=CHECK_NO_HW_DOT
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --cpu=cortex-a35 --emit-asm  -o - | FileCheck %s --check-prefix=CHECK_NO_HW_DOT
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --cpu=cortex-a53 --emit-asm  -o - | FileCheck %s --check-prefix=CHECK_NO_HW_DOT
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i8x16 --cpu=cortex-a53 --emit-asm  -o - | FileCheck %s --check-prefix=CHECK_NO_HW_DOT

// At -O0, the paths for instructions the CPU lacks must still be removed: no usdot without i8mm and no calls
// to the __not_supported stubs or to dot builtins.
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x4 --cpu=cortex-a55 -O0 --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=__not_supported \
// RUN:   --implicit-check-not=usdot '--implicit-check-not={{__dot[24]add_}}'
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i8x16 --cpu=cortex-a55 -O0 --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=__not_supported \
// RUN:   --implicit-check-not=usdot '--implicit-check-not={{__dot[24]add_}}'
// RUN: %{ispc} %s --arch=aarch64 --target=neon-i32x8 --cpu=cortex-a510 -O0 --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=__not_supported \
// RUN:   '--implicit-check-not={{__dot[24]add_}}'

// REQUIRES: ARM_ENABLED

// O0: {{^_?}}dot4add_u8i8___

// CHECK_ALL-LABEL: dot4add_u8u8___
// CHECK_ALL: udot

// CHECK_ALL-LABEL: dot4add_i8i8___
// CHECK_ALL: sdot

// CHECK_MIXED_SIGN-LABEL: dot4add_u8i8___
// CHECK_MIXED_SIGN: usdot

// Without i8mm, the mixed-sign forms are built from two sdot per vector of four 32-bit lanes: one vector for
// neon-i32x4, two for neon-i32x8 and neon-i16x8, and four for neon-i8x16. These functions are unmasked so that
// the body is not duplicated for the mask cases.
// CHECK_DOT_ONLY-LABEL: dot4add_u8i8___
// DOT_X4-COUNT-2: {{[[:space:]]}}sdot
// DOT_X8-COUNT-4: {{[[:space:]]}}sdot
// DOT_X16-COUNT-8: {{[[:space:]]}}sdot
// CHECK_DOT_ONLY-NOT: {{[[:space:]]}}sdot
// CHECK_DOT_ONLY-LABEL: dot4add_u8i8_sat___
// DOT_X4-COUNT-2: {{[[:space:]]}}sdot
// DOT_X8-COUNT-4: {{[[:space:]]}}sdot
// DOT_X16-COUNT-8: {{[[:space:]]}}sdot
// CHECK_DOT_ONLY-NOT: {{[[:space:]]}}sdot

// CHECK_MIXED_SIGN-LABEL: dot4add_u8i8_sat___
// CHECK_MIXED_SIGN: usdot

// The saturating forms add the plain dot product to acc with one saturation.
// CHECK_ALL-LABEL: dot4add_u8u8_sat___
// CHECK_ALL: udot
// CHECK_ALL-LABEL: dot4add_i8i8_sat___
// CHECK_ALL: {{[[:space:]]}}sdot

// CHECK_NO_HW_DOT-NOT: udot
// CHECK_NO_HW_DOT-NOT: sdot
// CHECK_NO_HW_DOT-NOT: usdot
void dot4add_u8u8(uniform uint a[], uniform uint b[], uniform uint dst[]) {
    dst[programIndex] = dot4add_u8u8packed(a[programIndex], b[programIndex], programIndex);
}

void dot4add_i8i8(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_i8i8packed(a[programIndex], b[programIndex], programIndex);
}

unmasked void dot4add_u8i8(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed(a[programIndex], b[programIndex], programIndex);
}

unmasked void dot4add_u8i8_sat(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_u8i8packed_sat(a[programIndex], b[programIndex], programIndex);
}

void dot4add_u8u8_sat(uniform uint a[], uniform uint b[], uniform uint dst[]) {
    dst[programIndex] = dot4add_u8u8packed_sat(a[programIndex], b[programIndex], programIndex);
}

void dot4add_i8i8_sat(uniform uint a[], uniform uint b[], uniform int dst[]) {
    dst[programIndex] = dot4add_i8i8packed_sat(a[programIndex], b[programIndex], programIndex);
}
