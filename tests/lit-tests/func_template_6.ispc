// Tests template with no params/return type i.e., void.
// Also test global symbol used in the template and print() expression.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -O0 -o - | FileCheck %s

int gVal;

template <typename T> noinline void goo() {
    print("\n Global val = % \n", (T)gVal);
}

// CHECK: define void @foo___
// CHECK: call void @goo___vyi___
// CHECK: call void @goo___vys___

// CHECK: define linkonce_odr void @goo___vyi___
// CHECK: define linkonce_odr void @goo___vys___
void foo() {
    goo<int>();
    goo<int16>();
}
