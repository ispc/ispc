// RUN: %{ispc} --target-os=windows --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s --implicit-check-not "llvm.dbg.cu"
// RUN: %{ispc} -g --target-os=windows --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_CODEVIEW
// RUN: %{ispc} --dwarf-version=3 --target-os=windows --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF
// RUN: %{ispc} -g --dwarf-version=3 --target-os=windows --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF
// RUN: %{ispc} --dwarf-version=3 -g --target-os=windows --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF
// RUN: %{ispc} --dwarf-version=3 --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF

// REQUIRES: WINDOWS_ENABLED
// CHECK_DWARF: Dwarf Version
// CHECK_CODEVIEW: CodeView
void foo() { return; }
