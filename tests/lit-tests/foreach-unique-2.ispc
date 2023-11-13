// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Can't assign to type "const uniform int32"

void foo(int x) {
    foreach_unique (u in x)
        ++u;
}


