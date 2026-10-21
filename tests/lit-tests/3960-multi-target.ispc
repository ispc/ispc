// Regression test for https://github.com/ispc/ispc/issues/3960
// Each target of a multi-target compilation describes forward declared and
// local structs in its own module.

// RUN: %{ispc} %s --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -g -O0 -o %t.o -h %t.h
// RUN: %{ispc} %s --target=sse4-i32x4,avx2-i32x8 --nostdlib --nowrap -g -O2 -o %t.o -h %t.h

// REQUIRES: X86_ENABLED

struct A;
uniform A *uniform global;
struct A {
    A *next;
    int val;
};

template <typename T> noinline T *uniform asPtr(void *uniform p) { return (T * uniform) p; }

export uniform int get(void *uniform v) {
    struct L;
    uniform L *uniform p = (uniform L * uniform)v;
    struct L {
        L *next;
        int l;
    };
    uniform L arr[2];
    return global->next->val + p->l + asPtr<uniform L>(v)->l + arr[1].l;
}
