// RUN: %{ispc} --target=host --nowrap %s -o t.o 2>&1 | FileCheck %s

// Functions below are called, so warnings should be generated.
__attribute__((deprecated)) void foo();
__attribute__((deprecated("reason"))) void foo1();
// CHECK: Warning: Function "foo()" is deprecated.
// CHECK: Warning: Function "foo1()" is deprecated: reason.
void bar() {
    foo();
    foo1();
}

// Functions below are not called, so no warnings should be generated.

// CHECK-NOT: Warning: Function "func()" is deprecated.
__attribute__((deprecated)) void func() { }

// CHECK-NOT: Warning: Function "func1()" is deprecated: reason.
__attribute__((deprecated("reason"))) void func1() { }

