// Regression test for https://github.com/ispc/ispc/issues/3960
// Debug info for structs whose definitions have errors, used by value and
// through pointers to their forward declarations, must not crash the compiler.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -g -O0 -o %t.o 2>&1 | FileCheck %s
// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -g -O2 -o %t.o 2>&1 | FileCheck %s

// CHECK-DAG: Error: Method declarations are not supported.
// CHECK-DAG: Error: "void" type illegal for struct member.
// CHECK-NOT: FATAL ERROR

struct A;
uniform A *uniform pointerA;
struct A {
    void foo();
};
uniform A valueA;

struct B;
uniform B *uniform pointerB;
struct B {
    void val;
};
uniform B valueB;

