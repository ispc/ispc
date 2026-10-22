// Check the parameter and return types of exported functions that reach a
// struct again through its own members.  A struct reached by value and again
// through a pointer is checked for both: A, returned by value, points to
// itself, so its SOA member is reached through a pointer, as when A is passed
// by pointer.  The const and non-const variants of a struct are checked
// separately, so the warning for the varying member of Q, reached through the
// const pointer in B, is reported before the error for B.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Error: --implicit-check-not="{{[0-9]+}}: Warning:"

struct S {
    int x;
};
struct A {
    uniform A *uniform next;
    soa<4> S data;
};

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: Warning: Exported function parameter "f" points to SOA type.
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to return a "varying" or vector type from exported function "f"
export uniform A f();

// CHECK: :[[@LINE+3]]:{{[0-9]+}}: Warning: Exported function parameter "p" points to SOA type.
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: Error: Varying pointer type parameter "p" is illegal in an exported function.
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Struct parameter "p" with vector typed member(s) is illegal in an exported function.
export void g(uniform A *uniform p) {}

struct Q;
struct B {
    const uniform Q *uniform q;
    uniform int *varying bad;
};
struct Q {
    uniform B *uniform b;
    varying int x;
};

// CHECK: :[[@LINE+3]]:{{[0-9]+}}: Warning: Exported function parameter "p" points to varying type
// CHECK: :[[@LINE+2]]:{{[0-9]+}}: Error: Varying pointer type parameter "p" is illegal in an exported function.
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Struct parameter "p" with vector typed member(s) is illegal in an exported function.
export void h(uniform Q *uniform p) {}
