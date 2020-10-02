// Test to check '#pragma nounroll' functionality for different loop statemnts.

// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap -O0 --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECKO0
// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap -O2 --emit-llvm-text -o - | FileCheck %s -check-prefix=CHECKO2
// REQUIRES: X86_ENABLED

// CHECKO0: br label %for_test, !llvm.loop

// CHECKO2: call void @goo_for___uni
// CHECKO2-NOT: call void @goo_for___uni

extern void goo_for(uniform int);
void foo_for() {
#pragma nounroll
    for (uniform int iter1 = 0; iter1 < 45; iter1++) {
        goo_for(iter1);
    }   
}


// CHECKO0: br i1 %less_iter1_load7_, label %do_loop, label %do_exit, !llvm.loop

// CHECKO2: call void @goo_do___uni
// CHECKO2-NOT: call void @goo_do___uni

extern void goo_do(uniform int);
void foo_do() {
    uniform int iter1 = 0;
#pragma nounroll
    do {
        goo_do(iter1);
        iter1++;
    } while(iter1 < 45);
}

// CHECKO0: br label %outer_not_in_extras, !llvm.loop

// CHECKO2: call void @goo_foreach___vyivyi
// CHECKO2: call void @goo_foreach___vyivyi
// CHECKO2-NOT: call void @goo_foreach___vyivyi

extern void goo_foreach(int, int);
void foo_foreach() {
#pragma nounroll
    foreach (iter1 = 0 ... 45, iter2 = 0 ... 86) {
        goo_foreach(iter1, iter2);
    }
}


// CHECKO0: br i1 %remaining_ne_zero, label %foreach_active_find_next, label %foreach_active_done, !llvm.loop

// CHECKO2: call void @goo_foreach_active___uni
// CHECKO2-NOT: call void @goo_foreach_active___uni

extern void goo_foreach_active(uniform int);
void foo_foreach_active() {
#pragma nounroll
    foreach_active (iter) {
        goo_foreach_active(iter);
    }
}


// CHECKO0:br i1 %remaining_ne_zero, label %foreach_find_next, label %foreach_done, !llvm.loop

// CHECKO2: call void @goo_foreach_unique___uni
// CHECKO2-NOT: call void @goo_foreach_unique___uni

extern void goo_foreach_unique(uniform int);
void foo_foreach_unique(int var) {
#pragma nounroll
    foreach_unique (iter in var) {
        goo_foreach_unique(iter);
    }
}


// CHECKO0:br label %outer_not_in_extras, !llvm.loop

// CHECKO2: call void @goo_foreach_tiled___vyivyivyi
// CHECKO2: call void @goo_foreach_tiled___vyivyivyi
// CHECKO2: call void @goo_foreach_tiled___vyivyivyi
// CHECKO2: call void @goo_foreach_tiled___vyivyivyi
// CHECKO2: call void @goo_foreach_tiled___vyivyivyi
// CHECKO2-NOT: call void @goo_foreach_tiled___vyivyivyi

extern void goo_foreach_tiled(int, int, int);
void foo_foreach_tiled() {
#pragma nounroll
    foreach_tiled (iter1 = 2 ... 14, iter2 = 24 ... 31, iter3 = 0 ... 8) {
        goo_foreach_tiled(iter1, iter2, iter3);
    }
}

// CHECKO0: !{!"llvm.loop.unroll.disable"}



