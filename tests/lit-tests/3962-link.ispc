// Check that template instantiations over different local structs with the
// same name in two translation units are not merged by the linker.
// See https://github.com/ispc/ispc/issues/3962

// RUN: %{ispc} --pic --target=host --nostdlib -O2 -DTU_A %s -o %t_a.o
// RUN: %{ispc} --pic --target=host --nostdlib -O2 -DTU_B %s -o %t_b.o
// RUN: %{cc} -x c -c %s -o %t.c.o
// RUN: %{cc} %t.c.o %t_a.o %t_b.o -o %t_ab.bin
// RUN: %{cc} %t.c.o %t_b.o %t_a.o -o %t_ba.bin
// RUN: %t_ab.bin | FileCheck %s
// RUN: %t_ba.bin | FileCheck %s

// REQUIRES: !MACOS_HOST

// CHECK: fa=11 fb=200 ga=21 gb=400

#ifdef ISPC
template <typename U> noinline uniform int64 rd(U *uniform s) { return s[1].x; }
template <typename U> noinline uniform int64 rdRef(uniform U &s) { return s.x; }

#ifdef TU_A
export uniform int64 fa(void *uniform v) {
    struct S {
        int x;
    };
    return rd((uniform S * uniform) v);
}
uniform struct {
    int x;
    int pad;
} anonA = {21, 0};
export uniform int64 ga() { return rdRef(anonA); }
#endif // TU_A

#ifdef TU_B
export uniform int64 fb(void *uniform v) {
    struct S {
        int64 x;
        double y;
    };
    return rd((uniform S * uniform) v);
}
uniform struct {
    int y;
    int x;
} anonB = {0, 400};
export uniform int64 gb() { return rdRef(anonB); }
#endif // TU_B
#else
#include <stdint.h>
#include <stdio.h>
int64_t fa(void *);
int64_t fb(void *);
int64_t ga(void);
int64_t gb(void);
int main() {
    // Large enough that the code for the other struct reads zeros in bounds.
    int a[8] = {10, 11, 12, 13, 0, 0, 0, 0};
    struct {
        int64_t x;
        double y;
    } b[2] = {{100, 0}, {200, 0}};
    printf("fa=%lld fb=%lld ga=%lld gb=%lld\n", (long long)fa(a), (long long)fb(b), (long long)ga(), (long long)gb());
    return 0;
}
#endif // ISPC
