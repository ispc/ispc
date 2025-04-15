// This test checks that there is no ICE on the following code.
// RUN: %{ispc} %s -o %t.o --target=host --nostdlib --nowrap 2>&1 | FileCheck %s

// CHECK-NOT: FATAL ERROR:
struct S {
    uint32 a;
};

S v;

void foo1(int i) {
    uint32 a = ((S*)(&v) + i)->a;
}

void foo2(int i) {
    uint32 a = (*((S*)(&v) + i)).a;
}
