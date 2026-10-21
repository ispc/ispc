// Resolving a forward-declared struct through a pointer to it must not change
// the layout of another struct with the same name.
// See https://github.com/ispc/ispc/issues/284 and
// https://github.com/ispc/ispc/issues/3959

// RUN: %{ispc} %s --target=avx2-i32x8 --nostdlib --nowrap -O2 --emit-llvm-text -o - | FileCheck %s

// REQUIRES: X86_ENABLED

// Comparison of a pointer to a local struct defined after the pointer.
// CHECK-LABEL: define {{.*}} @compare_first(
export uniform bool compare_first(void *uniform v) {
    struct L1;
    uniform L1 *uniform p = (uniform L1 *uniform)v;
    struct L1 { int x; };
    return p != NULL;
}

// CHECK-LABEL: define {{.*}} @compare_second(
// CHECK: getelementptr i8, ptr %v, i64 8
export uniform int64 compare_second(void *uniform v) {
    struct L1 { int64 y; };
    uniform L1 *uniform p = (uniform L1 *uniform)v;
    return (p + 1)->y;
}

// Template argument deduction from such a pointer.
template <typename T> uniform bool notNull(T p) { return p != NULL; }

// CHECK-LABEL: define {{.*}} @deduce_first(
export uniform bool deduce_first(void *uniform v) {
    struct L2;
    uniform L2 *uniform p = (uniform L2 *uniform)v;
    struct L2 { int x; };
    return notNull(p);
}

// CHECK-LABEL: define {{.*}} @deduce_second(
// CHECK: getelementptr i8, ptr %v, i64 8
export uniform int64 deduce_second(void *uniform v) {
    struct L2 { int64 y; };
    uniform L2 *uniform p = (uniform L2 *uniform)v;
    return (p + 1)->y;
}

// A pointer returned by a function declared before the struct is defined.
struct L3;
uniform L3 *uniform getL3();
struct L3 { int x; };

// CHECK-LABEL: define {{.*}} @return_first(
export uniform bool return_first() { return getL3() == NULL; }

// CHECK-LABEL: define {{.*}} @return_second(
// CHECK: getelementptr i8, ptr %v, i64 8
export uniform int64 return_second(void *uniform v) {
    struct L3 { int64 y; };
    uniform L3 *uniform p = (uniform L3 *uniform)v;
    return (p + 1)->y;
}
