// RUN: %{ispc} %s -MMM %t.dep -h %t.h
// RUN: FileCheck %s --input-file=%t.dep
// CHECK-NOT: stdlib.ispc
int no_stdlib_dep(int a, int b) {
    return a + b;
}