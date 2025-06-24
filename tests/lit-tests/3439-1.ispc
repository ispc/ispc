// This test checks that the compiler reports errors when setting the result of
// a function call returning a pointer (which must be considered as a rvalue).
// It also must not crash when trying to increment the result either (but just
// properly reports an error).

// RUN: not %{ispc} --target=host --emit-llvm-text --nowrap -g %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Left hand side of assignment expression can't be assigned to
// CHECK: Error: Can't assign to left-hand side of expression
// CHECK: Error: Can't post-increment non-lvalues
// CHECK: Error: Can't pre-increment non-lvalues
// CHECK-NOT: FATAL ERROR:

uniform int val = 0;
uniform int val2 = 1;

static uniform int *uniform foo() {
    return &val;
}

static void bar() {
    foo() = &val2;
}

static void bar2() {
    foo() += 1;
}

static void bar3() {
    foo()++;
}

static void bar4() {
    ++foo();
}

extern "C" uniform int main() {
    bar();
    return val;
}
