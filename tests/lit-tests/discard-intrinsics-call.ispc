// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED
// REQUIRES: ASAN_RUN

// CHECK-NOT: detected memory leaks

@llvm.x86.avx.addsub.ps.256;
