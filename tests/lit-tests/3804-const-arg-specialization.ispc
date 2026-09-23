// #3804: passing a `uniform const` arg into a helper that branches on it (the
// common "compile-time variant" idiom) must still be specialized away after the
// inlining changes. Calling with literal true/false should fold each branch, so
// `run` ends up with two specialized loops (*2 and +100) and no runtime select.

// RUN: %{ispc} %s --target=avx2-i32x8 --nowrap -O2 --emit-llvm-text -o - \
// RUN:   | FileCheck %s --implicit-check-not="select"

// REQUIRES: X86_ENABLED

// Both helpers are fully inlined: no definition of or call to them remains.
// CHECK-NOT: @Calculate
// CHECK-NOT: @ProcessArray

// The constant methodA is folded inside run itself: one path multiplies by 2,
// the other adds 100.
// CHECK-LABEL: define void @run(
// CHECK-DAG: shl nsw <8 x i32>{{.*}}splat (i32 1)
// CHECK-DAG: add nsw <8 x i32>{{.*}}splat (i32 100)

static inline int Calculate(int value, uniform const bool methodA) {
    if (methodA)
        return value * 2;
    else
        return value + 100;
}

static void ProcessArray(uniform int values[], uniform int n, uniform const bool methodA) {
    foreach (i = 0 ... n)
        values[i] = Calculate(values[i], methodA);
}

export void run(uniform int values[], uniform int n, uniform const bool methodA) {
    if (methodA)
        ProcessArray(values, n, true);
    else
        ProcessArray(values, n, false);
}
