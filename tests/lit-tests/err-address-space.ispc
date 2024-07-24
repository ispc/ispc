// RUN: not %{ispc} --target=host --nowrap --nostdlib  %s -o t.o 2>&1 | FileCheck %s

#define ATTR __attribute__((address_space(1)))
#define ATTRN(N) __attribute__((address_space(N)))

// CHECK: Warning: "address_space" attribute is only allowed for pointer or reference types, "func_2".
ATTR void func_2();

// It is not yet supported to have __attribute__ in struct declaration.
// CHECK: Error: syntax error, unexpected '__attribute__'
struct S {
  ATTR void *ptr_str;
};

// CHECK: Warning: "address_space" attribute is only allowed for pointer or reference types, "x".
ATTRN(1) uniform int x;

// CHECK: Error: "address_space" attribute 20 is out of scope of supported [0, 4], "ptr_2".
ATTRN(20) uniform int * uniform ptr_2;

void func() {
    // CHECK: Warning: "address_space" attribute is only allowed for pointer or reference types, "l1".
    ATTRN(1) uniform int l1;

    // CHECK: Error: "address_space" attribute 20 is out of scope of supported [0, 4], "lptr".
    ATTRN(20) uniform int * uniform lptr;
}

// CHECK: Warning: "address_space" attribute is only allowed for pointer or reference types, "p1".
// CHECK: Error: "address_space" attribute 20 is out of scope of supported [0, 4], "pptr1".
void decl1(ATTRN(1) uniform int p1, ATTRN(20) uniform int * uniform pptr1);

// CHECK: Warning: "address_space" attribute is only allowed for pointer or reference types, "p2".
// CHECK: Error: "address_space" attribute 20 is out of scope of supported [0, 4], "pptr2".
void def1(ATTRN(1) uniform int p2, ATTRN(20) uniform int * uniform pptr2) { }

// CHECK: Error: Can't convert between pointer types with different address spaces "uniform int16 * uniform" and "uniform int16 addrspace(2) * uniform" for return statement.
ATTRN(2) uniform int16 *uniform func_3(uniform int16 *uniform x) { return x; }

// CHECK: Error: Can't convert from pointer type "uniform int16 addrspace(3) * uniform" to incompatible pointer type "uniform int8 * uniform" for return statement.
uniform int8 *uniform func_4(ATTRN(3) uniform int16 *uniform x) { return x; }

// CHECK: Error: Can't convert from reference type "uniform int8 addrspace(3) &" to incompatible reference type "uniform int8 &" for return statement.
uniform int8 &func_5(ATTRN(3) uniform int8 &x) { return x; }
