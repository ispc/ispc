// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: variable 't' has incomplete struct type 'varying struct Test' and cannot be defined
extern "C" struct Test;
export void simple() { Test t; }

// CHECK-NOT: FATAL ERROR:
