// Guards the location of performance warnings for gathers and scatters inside
// `inline` helpers: the warning points at the operation in the helper, it is
// printed once per source operation however often the helper is inlined, and
// it is not printed when the index becomes linear after inlining.
//
// Ideal: = today.
//
// At -O0 no gather or scatter warnings are printed today; that is not pinned
// here, because the intended -O0 behavior is not defined yet.

// RUN: %{ispc} %s --target=sse4-i32x4 --nostdlib --nowrap -o /dev/null 2> %t.sse4
// RUN: FileCheck %s --input-file=%t.sse4
// RUN: FileCheck %s --input-file=%t.sse4 --check-prefix=COUNT3
// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -o /dev/null 2> %t.avx2
// RUN: FileCheck %s --input-file=%t.avx2
// RUN: FileCheck %s --input-file=%t.avx2 --check-prefix=COUNT3
// RUN: %{ispc} %s --target=avx512skx-x16 --nostdlib --nowrap -o /dev/null 2> %t.avx512skx
// RUN: FileCheck %s --input-file=%t.avx512skx
// RUN: FileCheck %s --input-file=%t.avx512skx --check-prefix=COUNT3

// REQUIRES: X86_ENABLED

// The warning checks match in any order; the COUNT prefix checks that
// there are no other warnings.
// COUNT3-COUNT-3: Performance Warning
// COUNT3-NOT: Performance Warning

// Gather and scatter in helpers with one call site each.
inline float fetch(uniform float base[], int i) {
    // CHECK-DAG: perf-warn-inline-location.ispc:[[#@LINE+1]]:12: Performance Warning: Gather required to load value.
    return base[i];
}
inline void put(uniform float base[], int i, float v) {
    // CHECK-DAG: perf-warn-inline-location.ispc:[[#@LINE+1]]:5: Performance Warning: Scatter required to store value.
    base[i] = v;
}
void one_site(uniform float a[], uniform int idx[], uniform float o[]) {
    float v = fetch(a, idx[programIndex]);
    put(o, idx[programIndex], v);
}

// A helper with two call sites gets one warning.
inline float fetch_twice(uniform float base[], int i) {
    // CHECK-DAG: perf-warn-inline-location.ispc:[[#@LINE+1]]:12: Performance Warning: Gather required to load value.
    return base[i];
}
void two_sites(uniform float a[], uniform int idx[], uniform float o[]) {
    o[programIndex] = fetch_twice(a, idx[programIndex]) + fetch_twice(a, idx[programIndex] + 7);
}

// The index is linear after inlining: a vector load, no warning.
inline float fetch_linear(uniform float base[], int i) { return base[i]; }
void linear_after_inline(uniform float a[], uniform float o[]) { o[programIndex] = fetch_linear(a, programIndex + 3); }
