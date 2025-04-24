// RUN: %{ispc} --nowrap --emit-llvm-text -o - --target=host %s 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR

uniform int** f(uniform int** p) { }
