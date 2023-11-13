// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to provide "varying" allocation count with "uniform new" expression

int * func(int x) {
    return uniform new int[x];
}
