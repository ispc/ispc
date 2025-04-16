// This test checks that generated code does not try divide by zero

// RUN: %{ispc} --pic --target=host %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK-NOT: Floating point exception
// CHECK: Dst[0] = 0
// CHECK-NEXT: Dst[1] = 0
// CHECK-NEXT: Dst[2] = 0
// CHECK-NEXT: Dst[3] = 0
// CHECK-NEXT: Dst[4] = 0
// CHECK-NEXT: Dst[5] = 0
// CHECK-NEXT: Dst[6] = 0
// CHECK-NEXT: Dst[7] = 0
// CHECK-NEXT: Dst[8] = 1
// CHECK-NEXT: Dst[9] = 1
// CHECK-NEXT: Dst[10] = 2
// CHECK-NEXT: Dst[11] = 2
// CHECK-NEXT: Dst[12] = 4
// CHECK-NEXT: Dst[13] = 6
// CHECK-NEXT: Dst[14] = 14
// CHECK-NEXT: Dst[15] = 0

#ifdef ISPC
export void Divi(uniform int Dst[], uniform int Src0[], uniform int Src1[], uniform int Count) {
    foreach (i = 0 ... Count) {
        bool bDivByZero = Src1[i] == 0;
        Dst[i] = bDivByZero ? 0 : Src0[i] / Src1[i];
    }
}
#else
#include <stdio.h>

#define ITERATIONS 16

extern void Divi(int Dst[], int Src0[], int Src1[], int Count);

int main() {
    int Dst[ITERATIONS] = {0};
    int Src0[ITERATIONS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int Src1[ITERATIONS] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

    Divi(Dst, Src0, Src1, ITERATIONS);

    for (int i = 0; i < ITERATIONS; i++) {
        printf("Dst[%d] = %d\n", i, Dst[i]);
    }
}
#endif // ISPC
