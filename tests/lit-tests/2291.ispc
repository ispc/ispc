// This test checks that this code does not cause any crash (fatal error) 
// due to the address of the `dummy_error` being requested, even though this 
// variable is `const` and also set to a compile-time constant.

// RUN: %{ispc} %s --emit-llvm-text --nowrap --nostdlib --target=host -g -o - 2>&1 | FileCheck %s

// CHECK-NOT: Error:
// CHECK-NOT: FATAL ERROR:

uniform int foo(uniform const int8 * uniform bar);

void crash() {
  const uniform int8 dummy_error = 0;
  uniform int err = 0;
  err = foo(&dummy_error);
}
