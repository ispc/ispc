// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Type conversion from "varying struct P" to "varying unsigned int32" for item count is not possible

struct P { int x; };

int * func(P p) {
    return new int[p];
}
