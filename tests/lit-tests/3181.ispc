// Check that target-specific instructions are used for int8/int16 masked store on avx512
// RUN: %{ispc} --target=avx512skx-x8 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefix=CHECK_X8 
// RUN: %{ispc} --target=avx512skx-x16 --nowrap -O2 --emit-asm --x86-asm-syntax=intel %s -o - | FileCheck %s --check-prefix=CHECK_X16

// REQUIRES: X86_ENABLED && !MACOS_HOST

// CHECK-LABEL: update_int8
// CHECK_X16-COUNT-2: vmovdqu8
void update_int8(uniform int8 values[16], bool upd, uniform int8 newVal) {
    for (uniform int i = 0; i < 16; i += programCount) {
        int index = i + programIndex;
        values[index] = select(upd, newVal, values[index]);
    }
}
// CHECK-LABEL: update_int16
// CHECK_X8-COUNT-4: vmovdqu16
// CHECK_X16-COUNT-2: vmovdqu16
void update_int16(uniform int16 values[16], bool upd, uniform int16 newVal) {
    for (uniform int i = 0; i < 16; i += programCount) {
        int index = i + programIndex;
        values[index] = select(upd, newVal, values[index]);
    }
}