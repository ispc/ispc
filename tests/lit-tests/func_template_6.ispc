// Tests template with no parasm/return type i.e., void

// RUN: %{ispc}  %s --emit-llvm-text --target=host -O0 -o - | FileCheck %s

int gVal;

template <typename T> noinline void goo() {
    print("\n Global val = % \n", (T)gVal);
}

// CHECK: define void @foo___
// CHECK: call void @goovyi___
// CHECK: call void @goovys___

// CHECK: define void @goovyi___
// CHECK: define void @goovys___
void foo() {
    goo<int>();
    goo<int16>();
}
