// RUN: %{ispc} --help 2>&1 | FileCheck %s --check-prefix=CHECK-HELP
// RUN: %{ispc} --help-dev 2>&1 | FileCheck %s --check-prefix=CHECK-HELP-DEV

// CHECK: Warning: --nocpp is deprecated and will be removed in the future.

// CHECK-HELP-NOT: --nocpp
// CHECK-HELP-DEV: --nocpp
int foo() { return 42; }
