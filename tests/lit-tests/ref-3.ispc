// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: syntax error, unexpected '*',

void foo(int & * x) {
    *x = NULL;
}
