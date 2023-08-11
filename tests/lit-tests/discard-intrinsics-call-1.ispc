// RUN: not %{ispc} --target=host --nostdlib --nowrap -o - %s 2>&1 | FileCheck %s

// REQUIRES: X86_ENABLED

// CHECK: Error: syntax error
// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process

@llvm.x86.avx.addsub.ps.256;
