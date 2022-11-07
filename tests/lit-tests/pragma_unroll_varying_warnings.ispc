// Test to check '#pragma nounroll' functionality for different loop statemnts.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap 2>&1 | FileCheck %s

// CHECK: Warning: '#pragma unroll/nounroll' for varying for loop is slow. Try '#pragma unroll/nounroll' for foreach loop.
extern void goo_for_var(int);
void foo_for_var() {
#pragma unroll(3)
    for (int iter1 = 0; iter1 < 45; iter1++) {
        goo_for_var(iter1);
    }
}

// CHECK: Warning: '#pragma unroll/nounroll' ignored - not supported for varying do loop.
extern void goo_do_var(int);
void foo_do_var() {
    int iter1 = 0;
#pragma unroll(3)
    do {
        goo_do_var(iter1);
        iter1++;
    } while(iter1 < 45);
}

// CHECK: Warning: '#pragma unroll/nounroll' ignored - not supported for foreach_active loop.
extern void goo_foreach_active(uniform int);
void foo_foreach_active() {
#pragma unroll 3
    foreach_active (iter) {
        goo_foreach_active(iter);
    }
}

// CHECK: Warning: '#pragma unroll/nounroll' ignored - not supported for foreach_unique loop.
extern void goo_foreach_unique(uniform int);
void foo_foreach_unique(int var) {
#pragma unroll 3
    foreach_unique (iter in var) {
        goo_foreach_unique(iter);
    }
}

extern void goo_foreach(int, int);
void foo_foreach() {
#pragma unroll(3)
    foreach (iter1 = 0 ... 45, iter2 = 0 ... 86) {
        goo_foreach(iter1, iter2);
    }
}

extern void goo_foreach_tiled(int, int, int);
void foo_foreach_tiled() {
#pragma unroll 3
    foreach_tiled (iter1 = 2 ... 14, iter2 = 24 ... 31, iter3 = 0 ... 8) {
        goo_foreach_tiled(iter1, iter2, iter3);
    }
}
