// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "uniform" qualifier is illegal with "void" type

struct FFF {
    uniform void *x;
};
