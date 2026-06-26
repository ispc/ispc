// Unwind v3 (the "winx64-eh-unwind" module flag) is Windows-only, so on Linux
// it is never set even for APX-capable targets that enable the EGPR registers
// (R16-R31) by default. The Windows-side coverage lives in apx-win-unwindv3.ispc.

//; RUN: %{ispc} %s --target=avx10.2dmr-x16 --arch=x86-64 --target-os=linux --emit-llvm-text -o - | FileCheck %s --check-prefix=NO-V3

// REQUIRES: X86_ENABLED && LINUX_ENABLED && LLVM_23_0+

// NO-V3-NOT: winx64-eh-unwind

void foo(uniform int a[], uniform int n) {
    for (uniform int i = 0; i < n; ++i)
        a[i] = a[i] + 1;
}
