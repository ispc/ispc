// RUN: %{ispc} -E --debug-phase=0:last %s -o - 2>&1 | FileCheck %s

#ifndef ISPC
// CHECK-NOT: LLVM IR after phase
void foo() {}
#endif
