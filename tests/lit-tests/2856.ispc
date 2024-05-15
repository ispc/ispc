// RUN: not %{ispc} -o - -g ../ 2>&1 | FileCheck -check-prefix=CHECK-DIR %s
// RUN: not %{ispc} -o - ../ 2>&1 | FileCheck -check-prefix=CHECK-DIR %s

// CHECK-DIR-NOT: FATAL ERROR: Unhandled signal sent to process
// CHECK-DIR: Error: File {{.*}} is a directory.

void foo() { return; }
