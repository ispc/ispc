// This test checks x86 instructions generated when vectorcall calling convention is not used on Windows.
// Without vectorcall, 'varying' arguments are not passed via vector registers.
// This results in a 'mov' instruction to move the data to vector registers in addition to the vector 'add'.
// RUN: %{ispc}  %s --emit-asm --no-vectorcall --target=avx2-i32x8 --target-os=windows -o - | FileCheck %s -check-prefix=CHECK_AVX2
// RUN: %{ispc}  %s --emit-asm --no-vectorcall --target=sse2-i32x4 --target-os=windows -o - | FileCheck %s -check-prefix=CHECK_SSE2
// RUN: %{ispc}  %s --emit-asm --no-vectorcall --target=sse4-i16x8 --target-os=windows -o - | FileCheck %s -check-prefix=CHECK_SSE4
// RUN: %{ispc}  %s --emit-asm --no-vectorcall --target=avx512skx-x16 --target-os=windows -o - | FileCheck %s -check-prefix=CHECK_AVX512

// REQUIRES: X86_ENABLED

// REQUIRES: WINDOWS_ENABLED

// CHECK_AVX2: mov

// CHECK_SSE2: mov

// CHECK_SSE4: mov

// CHECK_AVX512: mov

int addVal( int val1, int val2) {
    return val1 + val2;
}
