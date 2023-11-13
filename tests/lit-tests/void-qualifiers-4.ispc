// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "varying" qualifier is illegal with "void" type

void foo(varying void *x) { }
