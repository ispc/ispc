// Check the order of the types that the header declares for structs that
// exported functions reach through cycles of forward declared structs.  The
// members of a struct reached again while they are being processed are
// processed again, so the uniform S and its enum E, reached through the
// members of B, come before the varying S, reached through the members of A,
// and the struct C is declared with the alignment of C64, which B reaches
// first.

// RUN: %{ispc} %s --target=host --nostdlib --nowrap -o %t.o -h %t.h
// RUN: FileCheck --input-file=%t.h %s
// RUN: %{cxx} -x c++ -fsyntax-only %t.h

// REQUIRES: !MACOS_HOST

// CHECK: enum E {
// CHECK: struct S {
// CHECK: __ISPC_ALIGNED_STRUCT__(64) C {
// CHECK: {{(struct|__ISPC_ALIGNED_STRUCT__\([0-9]+\))}} v{{[0-9]+}}_varying_S {

enum E { e };
struct A;
struct S {
    E value;
};
struct C {
    int c;
};
typedef C __attribute__((aligned(64))) C64;
struct B {
    uniform A *uniform a;
    uniform S *uniform s;
    uniform C64 *uniform c;
};
struct A {
    uniform B *uniform b;
    varying S *uniform s;
    uniform C *uniform c;
};
export void f(uniform B *uniform p) {}

