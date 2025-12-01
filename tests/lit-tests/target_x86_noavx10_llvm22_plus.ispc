// The test checks that avx10.2nvl-x* targets are not recognized before LLVM 22.

// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x4 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x8 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x16 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x32 2>&1 | FileCheck %s
// RUN: not %{ispc} %s -o %t.o --nostdlib --target=avx10.2nvl-x64 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED && !MACOS_HOST && LLVM_20_0+ && !LLVM_22_0+

// CHECK: Error: Incorrect targets: avx10.2nvl
uniform int j;

int foo(int i) { return i + 1; }
