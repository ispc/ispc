// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Multiple labels named "label" in function

void func(int x) {
 label:
    ;
 label:
     ;
}


