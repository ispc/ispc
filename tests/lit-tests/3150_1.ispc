// RUN: not %{ispc} --nocpp --nowrap --emit-llvm-text -o - --target=host %s 2>&1 | FileCheck %s

// XFAIL: *

// CHECK-NOT: Error: Assertion failed {{.*}} "maskedStoreFunc != nullptr"
int foo() { return 42; }
