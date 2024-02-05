// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

struct Foo { 
//CO    uniform int a; 
//CO    int a;
//CO    float *x;
//CO    float y[5], z; 
    double d[5];
    float x, y, z;
};

void foo3(uniform Foo uf) {
    soa<4> Foo xyz;

// CHECK: Error: Can't convert between types "uniform struct Foo" and "soa<4> struct Foo" with different SOA widths
    xyz= uf;

// CHECK: Error: Type conversion from "const uniform int[[W:[0-9]+]]" to "soa<4> struct Foo" for = is not possible
    xyz = 0;

// CHECK: Error: Can't convert between types "const uniform int[[W:[0-9]+]]" and "soa<4> float" with different SOA widths
    xyz.x = 0;

// CHECK: Error: Can't apply unary operator to SOA type "soa<4> struct Foo"
    ++xyz;

// CHECK: Error: Can't apply unary operator to SOA type "soa<4> float"
    -xyz.x;

// CHECK: Error: Illegal to use binary operator - with SOA type "soa<4> float"
    xyz.x = xyz.y-xyz.x;
}