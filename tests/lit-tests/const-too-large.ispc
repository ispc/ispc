// RUN: %{ispc} --target=host --nowrap --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s

// CHECK: Warning: Binary expression with type "const uniform int8" can't represent value.

int8 v = (int8)64 * (int8)32;
