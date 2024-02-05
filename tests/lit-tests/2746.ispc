
// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Unable to apply SOA conversion to struct due to "uniform int32" member "x" with bound "uniform" 
// CHECK-NOT: FATAL ERROR: Unhandled signal sent

struct Point8 { uniform int x; float a, b, c; };

void foo(soa<8> Point8 pts[]);
