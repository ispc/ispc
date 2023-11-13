// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Array dimension must be representable with a 32-bit integer.

struct foo {
    int x[0xffffffffffff];
};


