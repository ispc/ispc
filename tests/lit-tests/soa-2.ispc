// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: soa<3> width illegal. Value must be positive power of two

struct F { float a, b, c; };

soa<3> F farray[10];

