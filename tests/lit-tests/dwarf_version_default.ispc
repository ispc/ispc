// Test that the default DWARF version is 5 when -g is specified
// without an explicit --dwarf-version flag.

// RUN: %{ispc} -g --target=host --target-os=linux --nostdlib --nowrap --emit-llvm-text -o - %s 2>&1 | FileCheck %s

// REQUIRES: LINUX_ENABLED

// CHECK: Dwarf Version", i32 5

void foo() { return; }
