// RUN: %{ispc} %s --nowrap --target=host --emit-llvm-text --enable-llvm-intrinsics -o - 2>&1 | FileCheck %s

// CHECK: llvm.readcyclecounter
// CHECK-NOT: Error: syntax error

uniform int64 foo() { return @llvm.readcyclecounter(); }
