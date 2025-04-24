// This test checks that there is no ICE on the following code.

// RUN: %{ispc} --target=host --nowrap --nostdlib --emit-llvm-text --debug-phase=210:210 %s -o - | FileCheck %s

// CHECK-NOT: FATAL ERROR: Unhandled signal sent to process
struct S {
    uint32 a;
};

S v;

// CHECK-LABEL: @foo1
// CHECK: __pseudo_gather64_i32
// CHECK: __pseudo_masked_store_i32
uint32 foo1(int i) {
    return ((S*)(&v) + i)->a;
}
// CHECK-LABEL: @foo2
// CHECK: __pseudo_gather64_i32
// CHECK: __pseudo_masked_store_i32
uint32 foo2(int i) {
    return (*((S*)(&v) + i)).a;
}
