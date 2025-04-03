// The test checks that avx10.2-x* targets are not recognized before LLVM 20.

// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x4 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x8 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x16 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x32 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2-x64 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED && !MACOS_HOST && !LLVM_20_0+

// CHECK: Error: Incorrect targets: avx10.2
uniform int j;

int foo(int i) { return i + 1; }
