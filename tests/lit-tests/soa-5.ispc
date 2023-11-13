// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "uniform" qualifier and "soa<4>" qualifier can't both be used

struct F { float a, b, c; };

uniform soa<4> F farray[10];

