// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Member operator "->" can't be used with expression of "uniform int32" type

int func(int *a) {
    a->x = 0;
}
