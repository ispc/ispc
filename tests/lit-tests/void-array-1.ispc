// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Arrays of "void" type are illegal

float f_fu(uniform void aFOO[]) {
    return 0;
}
