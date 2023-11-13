// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Type conversion from "const uniform int32" to "uniform int32 * varying" for initializer is not possible

int voo() {
    int * varying foo = 1l;
}
