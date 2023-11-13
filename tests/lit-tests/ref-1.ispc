// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: syntax error, unexpected '&'

int foo(int & & bar) {
    bar = 0;
    return bar;
}
