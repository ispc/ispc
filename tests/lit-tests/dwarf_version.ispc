// RUN: %{ispc} --dwarf-version=2 --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF_V2
// RUN: %{ispc} --dwarf-version=3 --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF_V3
// RUN: %{ispc} --dwarf-version=4 --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF_V4
// RUN: %{ispc} --dwarf-version=5 --target=host --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK_DWARF_V5

// CHECK_DWARF_V2: Dwarf Version", i32 2
// CHECK_DWARF_V3: Dwarf Version", i32 3
// CHECK_DWARF_V4: Dwarf Version", i32 4
// CHECK_DWARF_V5: Dwarf Version", i32 5
void foo() { return; }

