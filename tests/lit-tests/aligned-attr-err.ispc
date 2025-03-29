// Check aligned attribute incorrect usage.

// RUN: not %{ispc} --nowrap --target=host --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// CHECK: Error: syntax error, unexpected '-'.
__attribute__((aligned(-2))) uniform int x1;

// CHECK: Error: Please provide an alignment value
__attribute__((aligned(0))) uniform int x2;

// CHECK: Error: Please provide an alignment value
__attribute__((aligned)) uniform int x3;

// CHECK: Error: requested alignment is too large.
__attribute__((aligned(4294967296))) uniform int x4;

// CHECK-COUNT-2: Error: requested alignment is not a power of 2.
__attribute__((aligned(4294967295))) uniform int x5;
__attribute__((aligned(6))) uniform int x5;
