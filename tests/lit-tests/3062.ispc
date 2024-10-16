// RUN: %{ispc} --target=host %s -M -MT target_name -MF %t.d 2>&1 | FileCheck %s
// RUN: %{ispc} --target=host %s -M -MT target_name 2>&1 | FileCheck %s --check-prefix=INMEM_HEADER

// CHECK-NOT: Warning: Emitting dependencies file, but filename {{.*}}
// INMEM_HEADER-NOT: ^ /core.isph{{.*}}
// INMEM_HEADER-NOT: ^ /stdlib.isph{{.*}}

__attribute__((unknown)) void foo() {}
