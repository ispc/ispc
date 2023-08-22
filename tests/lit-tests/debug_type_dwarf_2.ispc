// RUN: %{ispc} --target-os=linux --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --implicit-check-not "llvm.dbg.cu"
// RUN: %{ispc} -g --target-os=linux --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF
// RUN: %{ispc} -g --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF

// REQUIRES: LINUX_HOST
// CHECK_DWARF: Dwarf Version
void foo() { return; }
