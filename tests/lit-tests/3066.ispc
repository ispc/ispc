// RUN: %{ispc} --target=host --target-os=windows %s -M -MT target_name 2>&1 | FileCheck %s

// REQUIRES: WINDOWS_ENABLED

// CHECK-NOT: ^ c:/core.isph{{.*}}
// CHECK-NOT: ^ c:/stdlib.isph{{.*}}

void foo() {}
