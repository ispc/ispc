// Regression test for https://github.com/ispc/ispc/issues/284
// The parameter and return types of exported functions are checked once the
// forward declared structs they reach are defined later in the file, as they
// are when the structs are defined before the functions.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Error: --implicit-check-not=Warning:

struct VaryingPtr;
struct VaryingMember;
struct Indirect;
struct Array;
struct Undefined;

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: Error: Varying pointer type parameter "p" is illegal in an exported function.
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Struct parameter "p" with vector typed member(s) is illegal in an exported function.
export void ptrMember(uniform VaryingPtr *uniform p) {}

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to return a "varying" or vector type from exported function "ret"
export uniform VaryingMember ret();

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Warning: Exported function parameter "p" points to varying type
export void indirect(uniform Indirect *uniform p) {}

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: Error: Varying pointer type parameter "p" is illegal in an exported function.
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Varying parameter "p" is illegal in an exported function.
export void array(uniform Array (*uniform p)[2]) {}

// Not defined anywhere: nothing to check.
export void undefined(uniform Undefined *uniform p) {}

struct VaryingPtr {
    uniform int *varying q;
};

struct VaryingMember {
    varying int a;
};

struct Indirect {
    uniform Indirect *uniform next;
    uniform VaryingMember *uniform m;
};

struct Array {
    uniform int *varying q;
};

export uniform VaryingMember ret() {
    uniform VaryingMember v;
    return v;
}
