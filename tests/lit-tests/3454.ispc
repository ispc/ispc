// This test checks that alloca is not a reserved keyword of the language.

// RUN: %{ispc} --target=host --emit-llvm-text --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: Error:
// CHECK-NOT: FATAL ERROR:

int test1() {
    int alloca = 42;
    return alloca;
}

int test2() {
    void* addr = alloca(1000);
}
