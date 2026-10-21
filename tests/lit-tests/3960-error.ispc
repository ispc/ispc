// Regression test for https://github.com/ispc/ispc/issues/3960
// Invalid struct members are reported when the struct is declared, so the
// diagnostics don't depend on whether debug info for the struct is emitted.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -O0 -o %t.o 2>&1 | FileCheck %s
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -g -O0 -o %t.o 2>&1 | FileCheck %s

// CHECK-DAG: 3960-error.ispc:[[#@LINE+8]]:{{[0-9]+}}: Error: Method declarations are not supported.
// CHECK-DAG: 3960-error.ispc:[[#@LINE+12]]:{{[0-9]+}}: Error: Method declarations are not supported.
// CHECK-DAG: 3960-error.ispc:[[#@LINE+16]]:{{[0-9]+}}: Error: Method declarations are not supported.
// CHECK-NOT: FATAL ERROR

struct A;
uniform A *uniform globalA;
struct A {
    void foo();
};

typedef void F();
struct B {
    F method;
};

// Not used at all.
struct C {
    void bar();
};
