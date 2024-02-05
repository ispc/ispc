// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Unable to find any matching overload for call to function "foo"

void foo();

void bar(int x) {
    foo(x);
}


// CHECK: Error: Unable to find any matching overload for call to function "foo2"

void foo2(int x);

void bar2(int x) {
    foo2();
}

// CHECK: Error: Unable to find any matching overload for call to function "foo3"

void foo3(int x);

void bar3(int x) {
    foo3(x, x);
}

// CHECK: Error: Ignoring redeclaration of symbol "a4"

int foo(int a4[], int a4) {
    return a4[0];
}

// CHECK: Error: Storage class "static" is illegal in function parameter declaration for parameter "x5"

void foo5(static int x5) {
}
