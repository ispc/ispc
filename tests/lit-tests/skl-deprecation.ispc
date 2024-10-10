// RUN: %{ispc} --nowrap --target=gen9-x8 --emit-spirv -o %t.spv %s 2>&1 | FileCheck %s --check-prefix=CHECK_GEN9-X8
// RUN: %{ispc} --nowrap --target=gen9-x16 --emit-spirv -o %t.spv %s 2>&1 | FileCheck %s --check-prefix=CHECK_GEN9-X16

// REQUIRES: XE_ENABLED

// CHECK_GEN9-X8: Warning: The target gen9-x8 is deprecated and will be removed in the future.
// CHECK_GEN9-X16: Warning: The target gen9-x16 is deprecated and will be removed in the future.

void foo() {}
