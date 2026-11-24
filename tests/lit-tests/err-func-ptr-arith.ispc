// Arithmetic on a function pointer, and indexing it, has no element size:
// it is an error, as for a void pointer, instead of a crash of the compiler.

// RUN: not %{ispc} %s --target=host --nostdlib --nowrap -o - 2>&1 | FileCheck %s --implicit-check-not="FATAL ERROR"

typedef float (*Fn)(float);

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to perform pointer arithmetic on function pointer type
int diff(Fn a, Fn b) { return a - b; }

// CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to perform pointer arithmetic on function pointer type
Fn next(Fn a) { return a + 1; }

void step(Fn a) {
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to perform pointer arithmetic on function pointer type
    a += 1;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to pre/post increment
    a++;
    // CHECK: :[[@LINE+1]]:{{[0-9]+}}: Error: Illegal to index into function pointer type
    a[1];
}
