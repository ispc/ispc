// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Unsized arrays aren't allowed except for the last member in a struct definition.

struct Foo6 {
    float a[];
    float b[10];
};

Foo6 f;

void Func6() {
    Foo6 f;
}

// CHECK: Error: Struct member "member_a" has same name as a previously-declared member

struct StrA {
    int member_a, member_a;
};

// CHECK: Error: Gather operation is impossible due to the presence of struct member "x" with uniform type

struct Foo1 { uniform int x; };

void foo1(uniform Foo1 array[], int index) {
    Foo1 v = array[index];
}

// CHECK: Error: Gather operation is impossible due to the presence of struct member "x" with uniform type

struct Foo2 { uniform int x; };

void a(uniform Foo2 * uniform array, int index) {
    Foo2 v = array[index];
}

// CHECK: Error: Gather operation is impossible due to the presence of struct member "x" with uniform type

struct Foo3 { uniform int x; };

void a(uniform Foo3 * varying array, int index) {
    Foo3 v = array[index];
}

// CHECK: Error: Gather operation is impossible due to the presence of struct member "x" with uniform type

struct Bar4 { uniform int x; };
struct Foo4 { varying Bar4 b; };

void a(uniform Foo4 * varying array, int index) {
    Foo4 v = array[index];
}

// CHECK: Error: Gather operation is impossible due to the presence of struct member "x" with uniform type

struct Bar5 { uniform int x; };
struct Foo5 { Bar5 b; };

void a(uniform Foo5 * varying array, int index) {
    Foo5 v = array[index];
}