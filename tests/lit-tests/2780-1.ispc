// RUN: %{ispc} --pic --target=avx512skx-x8 -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s

// REQUIRES: !MACOS_HOST && X86_ENABLED

// The result of round stdlib function should not depend on MXCSR
// CHECK: rounded 0: [2.000000,
// CHECK: rounded 1: [2.000000,

#ifdef ISPC
extern "C" void RoundToInt(uniform int i, uniform float input[])
{
    float r = round(input[programIndex]);
    print("rounded %: %\n", i, r);
}
#else
#include <xmmintrin.h>

extern void RoundToInt(int, float*);

int main() {
    float input[64] = { 2.5, 0.0 };
    unsigned int currentMXCSR, newMXCSR;
    currentMXCSR = _mm_getcsr();

    // Set rounding mode to round toward zero (01)
    newMXCSR = (currentMXCSR & ~0x6000) | (0x1 << 13);
    _mm_setcsr(newMXCSR);

    RoundToInt(0, input);

    // Set rounding mode to round to nearest (00), which is the default
    newMXCSR = (currentMXCSR & ~0x6000);
    _mm_setcsr(newMXCSR);

    RoundToInt(1, input);

    _mm_setcsr(currentMXCSR);
    return 0;
}
#endif // ISPC
