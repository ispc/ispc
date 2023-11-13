// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Storage class "static" is illegal in function parameter declaration for parameter "x"

void foo(static int x) {
}


