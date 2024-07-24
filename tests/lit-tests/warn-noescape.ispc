// RUN: %{ispc} --target=host --nowrap --nostdlib %s -o t.o 2>&1 | FileCheck %s

// CHECK: Warning: Ignoring "noescape" attribute for global variable "d".
__attribute__((noescape)) double uniform *uniform d;

// CHECK: Warning: Ignoring "noescape" attribute for global variable "x".
__attribute__((noescape)) int32 x;

void func_2() {
  // CHECK: Warning: Ignoring "noescape" attribute for variable "p".
  __attribute__((noescape)) uniform float * uniform p;
}

