// RUN: not %{ispc} --nowrap --target=host %s -o - --emit-llvm-text 2>&1 | FileCheck %s

#define MEM(A) __attribute__((memory(#A)))

// CHECK: Unknown memory attribute "WRITE".
MEM(WRITE) void bar();

// CHECK: Unknown memory attribute "READ".
MEM(READ) void foo();

// CHECK: Unknown memory attribute "NONE".
MEM(NONE) void foo1();
