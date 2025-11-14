// Test that --no-internal-export-functions generates only external versions
// of exported functions (without mask parameter). When this flag is used,
// exported functions cannot be called from ISPC code (would cause an error).

// RUN: %{ispc} %s --target=host --nowrap --nostdlib --no-internal-export-functions --emit-llvm-text -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} @exported_func(
// CHECK-NOT: define {{.*}} @exported_func___

export void exported_func() {
    // Empty function for testing
}

// Test that with default behavior (flag enabled), both internal and external versions are generated
// RUN: %{ispc} %s --target=host --nowrap --nostdlib --emit-llvm-text -o - 2>&1 | FileCheck %s -check-prefix=CHECK_INTERNAL

// CHECK_INTERNAL: define {{.*}} @exported_func___
// CHECK_INTERNAL: define {{.*}} @exported_func(
