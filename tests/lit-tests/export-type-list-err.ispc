// Check that a type export list with a type that is invalid after an error
// doesn't crash the compiler, wherever the invalid type is in the list, and
// that the other types in the list are still checked.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o %t.o 2>&1 | FileCheck %s --implicit-check-not=Error: --implicit-check-not="FATAL ERROR"

typedef int I;
struct S {
    int a;
};

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Type "I" is not a struct type! (/*unbound*/ int32)
export { S, struct I };

// CHECK: :[[@LINE+2]]:{{[0-9]+}}: Error: Type "I" is not a struct type! (/*unbound*/ int32)
// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Only struct, vector, and enum types, not "/*unbound*/ int32", are allowed in type export lists.
export { struct I, S, I };
