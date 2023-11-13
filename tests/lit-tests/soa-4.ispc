// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: syntax error, unexpected '-', expecting int

struct F { float a, b, c; };

soa<-4> F farray[10];

