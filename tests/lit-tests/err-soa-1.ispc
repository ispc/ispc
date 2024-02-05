// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Illegal to provide soa<4> qualifier with non-struct type "void"
soa<4> void foo1() { }

struct F { float a, b, c; };

// CHECK: Error: soa<3> width illegal. Value must be positive power of two
soa<3> F farray1[10];

// CHECK: Error: syntax error, unexpected '-', expecting int
soa<-4> F farray2[10];

// CHECK: Error: "uniform" qualifier and "soa<4>" qualifier can't both be used
uniform soa<4> F farray3[10];

// CHECK: Error: Can't convert between types "uniform struct F" and "soa<4> struct F" with different SOA widths
void foo(soa<8> F pts[]) {
    soa<4> F x = pts[0];
}

// CHECK: Error: Can't convert from pointer to SOA type "soa<8> struct A9 * uniform" to pointer to non-SOA type "void * varying" 
struct A9 { float a, b; };
soa<8> A9 as[100];
void foo9() {
    void *ptr = &as[0];
}