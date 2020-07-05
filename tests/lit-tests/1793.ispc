// RUN: %{ispc} --emit-asm --target=avx2-i32x8 --nostdlib %s -o - | FileCheck %s
// REQUIRES: X86_ENABLED
// REQUIRES: LLVM_10_0+

// This test is intended to check if the patch for #1793 is applied.
// The code below is dead code, but LLVM is not able to figure this out,
// so it just required to produce correct code.
// The test checks that "good" conde sequence is generated.
// As it's dead code, removing all instructions is also as valid code (but it's not checked).

extern uniform bool var_14;
extern uniform unsigned int64 arr_68;

// CHECK: xorl
// CHECH-NEXT: testb
unmasked extern void test() {
    if ((varying int8) var_14 < 0)
        arr_68 = 1;
}
