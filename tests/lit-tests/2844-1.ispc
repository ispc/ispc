// Check that we catch the case where a function returns an incomplete struct type

// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Error: return type is an incomplete type: varying struct Sx
struct Sx bar() {}

// CHECK-NOT: Error: return type is an incomplete type: varying struct Sy
struct Sy foo();

// CHECK-NOT: FATAL ERROR:
