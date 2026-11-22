// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o t.o 2>&1 | FileCheck %s

// Test that returning the address of a local variable or parameter issues a
// warning, since the memory it points to is invalid after the function returns.

uniform int g;

// CHECK: Warning: Returning address of local variable or parameter "x".
export uniform int *uniform ret_addr_of_local() {
    uniform int x = 0;
    return &x;
}

// CHECK: Warning: Returning address of local variable or parameter "x".
export uniform int *uniform ret_cast_addr_of_local() {
    uniform int x = 0;
    return (uniform int *uniform)&x;
}

// CHECK: Warning: Returning address of local variable or parameter "a".
export uniform int *uniform ret_array_decay() {
    uniform int a[4];
    return a;
}

// CHECK: Warning: Returning address of local variable or parameter "p".
export uniform int *uniform ret_addr_of_param(uniform int p) {
    return &p;
}

// CHECK: Warning: Returning address of local variable or parameter "x".
uniform int &ret_ref_to_local() {
    uniform int x = 0;
    return x;
}

// The following are all fine and must not warn:

// CHECK-NOT: Warning

// Returning the address of a global is fine.
export uniform int *uniform ret_addr_of_global() {
    return &g;
}

// Returning the address of a static local is fine (static storage duration).
export uniform int *uniform ret_addr_of_static_local() {
    static int s = 0;
    return &s;
}

// Returning a copy of a local pointer's value is fine.
export uniform int *uniform ret_ptr_value(uniform int *uniform p) {
    return p;
}

// Returning a copy of a local's value is fine.
export uniform int ret_value() {
    int x = 5;
    return x;
}

// Returning a constant is fine.
export uniform int *uniform ret_const() {
    return (uniform int *uniform)0;
}

// Returning a reference parameter is fine (it aliases caller-owned memory).
uniform int &ret_ref_param(uniform int &r) {
    return r;
}
