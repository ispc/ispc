// Check that template instantiations over a global anonymous enum named by a
// typedef, or over a non-type template parameter of its type, are merged by
// the linker, so a function-local static in them is shared by two translation
// units, as for named enums.

// RUN: %{ispc} --pic --target=host --nostdlib -O2 -DTU_A %s -o %t_a.o
// RUN: %{ispc} --pic --target=host --nostdlib -O2 -DTU_B %s -o %t_b.o
// RUN: %{cc} -x c -c %s -o %t.c.o
// RUN: %{cc} %t.c.o %t_a.o %t_b.o -o %t.bin
// RUN: %t.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: ta=1 tb=2 va=1 vb=2

#ifdef ISPC
typedef enum { ModeA = 7 } Mode;

template <typename T> noinline uniform int nextType() {
    static uniform int counter = 0;
    return ++counter;
}

template <Mode V> noinline uniform int nextValue() {
    static uniform int counter = 0;
    return ++counter;
}

#ifdef TU_A
export uniform int ta() { return nextType<Mode>(); }
export uniform int va() { return nextValue<ModeA>(); }
#endif // TU_A

#ifdef TU_B
export uniform int tb() { return nextType<Mode>(); }
export uniform int vb() { return nextValue<ModeA>(); }
#endif // TU_B
#else
#include <stdio.h>
int ta(void);
int tb(void);
int va(void);
int vb(void);
int main() {
    int a = ta();
    int b = tb();
    int c = va();
    int d = vb();
    printf("ta=%d tb=%d va=%d vb=%d\n", a, b, c, d);
    return 0;
}
#endif // ISPC
