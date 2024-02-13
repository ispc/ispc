// RUN: %{ispc} -DISPC --pic --target=host -h %t.h %s -o %t.o
// RUN: %{cc} -x c -c %s -o %t.c.o --include %t.h
// RUN: %{cc} %t.o %t.c.o -o %t.c.bin
// RUN: %t.c.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: a[2]=-3 b[2]=3

#ifdef ISPC
export uniform int func(uniform bool cond, uniform int a[], uniform int b[]) {
    return (cond ? a : b)[2];
}
#else
#include <stdio.h>
int main(int argc, char **argv) {
    int a[] = { -1, -2, -3, -4 };
    int b[] = { 1, 2, 3, 4 };
    int a2 = func(true, a, b);
    int b2 = func(false, a, b);
    printf("a[2]=%i b[2]=%i\n", a2, b2);
    return 0;
}
#endif
