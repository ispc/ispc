// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Array dimension must be non-negative

struct foo {
    int x[-1];
};


