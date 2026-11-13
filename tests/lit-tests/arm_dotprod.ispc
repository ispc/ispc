// Test checks emitted code for ARM (aarch32) dot product instructions.

// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --cpu=cortex-a55 --emit-asm  -o - | FileCheck %s --check-prefixes=CHECK,X4
// RUN: %{ispc} %s --arch=arm --target=neon-i32x8 --cpu=cortex-a55 --emit-asm  -o - | FileCheck %s --check-prefixes=CHECK,X8

// At -O0, the paths for instructions the CPU lacks must still be removed: no vusdot without i8mm and no calls
// to the __not_supported stubs or to dot builtins.
// RUN: %{ispc} %s --arch=arm --target=neon-i32x4 --cpu=cortex-a55 -O0 --emit-asm -o - \
// RUN:   | FileCheck %s --check-prefix=O0 --implicit-check-not=__not_supported \
// RUN:   --implicit-check-not=vusdot '--implicit-check-not={{__dot[24]add_}}'

// REQUIRES: ARM32_ENABLED && LINUX_HOST

// O0: {{^}}dot4add_u8i8___

// The mnemonics are anchored so that vusdot cannot match them.

// CHECK-LABEL: dot4add_u8u8___
// CHECK: {{[[:space:]]}}vudot.u8

// CHECK-LABEL: dot4add_i8i8___
// CHECK: {{[[:space:]]}}vsdot.s8

// Without i8mm, the mixed-sign form is built from two sdot per vector: one vector for neon-i32x4 and two for
// neon-i32x8. The function is unmasked so that the body is not duplicated for the mask cases.
// CHECK-LABEL: dot4add_u8i8___
// X4-COUNT-2: {{[[:space:]]}}vsdot.s8
// X8-COUNT-4: {{[[:space:]]}}vsdot.s8
// CHECK-NOT: {{[[:space:]]}}vsdot.s8
// CHECK-LABEL: dot4add_u8i8_sat___
// X4-COUNT-2: {{[[:space:]]}}vsdot.s8
// X8-COUNT-4: {{[[:space:]]}}vsdot.s8
// CHECK-NOT: {{[[:space:]]}}vsdot.s8

// The saturating forms add the plain dot product to acc with one saturation.
// CHECK-LABEL: dot4add_u8u8_sat___
// CHECK: {{[[:space:]]}}vudot.u8
// CHECK-LABEL: dot4add_i8i8_sat___
// CHECK: {{[[:space:]]}}vsdot.s8

void dot4add_u8u8(uniform int a[], uniform int b[], uniform uint dst[]) {
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
