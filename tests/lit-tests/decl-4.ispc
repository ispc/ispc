// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Illegal to declare an unsized array variable without providing an initializer expression to set its size

int func() {
    int a[];
}
