// RUN: %{ispc} --target=host --nowrap --nostdlib %s --emit-llvm-text -o - 2>&1 | FileCheck %s

// CHECK: Warning: Symbol "a" shadows symbol declared in outer scope

void foo() {
    int a;
    {
        int a;
    }
}
