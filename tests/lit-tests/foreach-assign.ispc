// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type "const varying int32" 

int foo() {
    foreach (i = 0 ... 10) {
        ++i;
    }
}
