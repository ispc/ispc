// RUN: %{ispc} --target=host --nowrap %s -o t.o 2>&1 | FileCheck %s

// Check that the deprecated attribute is working correctly for overloaded functions.

__attribute__((deprecated)) void foo(int x);
void foo(float x);
// CHECK: Warning: Function "foo(varying int32)" is deprecated.
void use_foo(int x) {
    foo(x);
}

__attribute__((deprecated)) void bar(int x);
void bar(float x);
// CHECK-NOT: Warning: Function "bar(varying float)" is deprecated.
void use_bar(float x) {
    bar(x);
}
