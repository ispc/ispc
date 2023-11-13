// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Parameter with type "void" illegal in function parameter list

void func(void x, void y) {
    return x+y;
}
