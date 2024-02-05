// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Member operator "." can't be used with expression of "uniform int32" type

int func(int *a) {
    a.x = 0;
}

// CHECK: Error: Member operator "->" can't be used with expression of "uniform int32" type

int foo(int *a) {
    a->x = 0;
}

// CHECK: Error: Member operator "." can't be applied to pointer type

struct Foo { int x; };

int func1(Foo *a) {
    a.x = 0;
}

// CHECK: Error: Member operator "->" can't be applied to non-pointer type "varying struct Foo3"

struct Foo3 { int x; };

int func3(Foo3 a) {
    a->x = 0;
}

// CHECK: Error: Illegal to dereference non-pointer type "varying float"

float func4(float a) {
    *a = 0;
    return 0;
}