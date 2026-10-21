// Regression test for https://github.com/ispc/ispc/issues/3960
// Invalid struct members are reported when the struct is declared, so the
// diagnostics don't depend on whether debug info for the struct is emitted,
// and debug info for structs whose definitions have errors, used by value and
// through pointers to their forward declarations, doesn't crash the compiler.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -O0 -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Error: --implicit-check-not=FATAL
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -g -O0 -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Error: --implicit-check-not=FATAL
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -g -O2 -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Error: --implicit-check-not=FATAL

struct A;
uniform A *uniform pointerA;
struct A {
    // CHECK-DAG: 3960-error.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Method declarations are not supported.
    void foo();
};
uniform A valueA;

struct B;
uniform B *uniform pointerB;
struct B {
    // CHECK-DAG: 3960-error.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: "void" type illegal for struct member.
    void val;
};
uniform B valueB;

typedef void F();
struct M {
    // CHECK-DAG: 3960-error.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Method declarations are not supported.
    F method;
};

// Not used at all.
struct C {
    // CHECK-DAG: 3960-error.ispc:[[#@LINE+1]]:{{[0-9]+}}: Error: Method declarations are not supported.
    void bar();
};
