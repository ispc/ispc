// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Unable to apply SOA conversion to struct due to "uniform int32" member "x" with bound "uniform" 

struct Point { uniform int x; float a, b, c; };

void foo(soa<8> Point pts[]);
