// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: "void" type variable illegal in declaration

int func() {
    void x;
}
