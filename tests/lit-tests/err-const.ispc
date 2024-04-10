// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Can't assign to type "const varying int32" on left-hand

int func() {
    const int x = 2;
    ++x;
}


// CHECK: Error: Can't assign to type "const varying unsigned int32" on left-hand

int func1() {
    const uint x = 2;
    x = 0;
}

// CHECK: Error: Can't assign to type "const varying int16" on left-hand

struct Foo {
    int16 x;
};

int func2(const Foo f) {
    f.x = 0;
}

// CHECK: Error: Can't assign to type "const varying int8" on left-hand

int func3(const int8 f) {
    f -= 2;
}

// CHECK: Error: Can't assign to type "const varying int64" on left-hand

int func4() {
    const int64 a[10] = {1,2,3,4,5,6,7,8,9,10};
    a[0] = 1;
}

// CHECK: Error: Can't assign to type "const varying unsigned int8" on left-hand

const unsigned int8 gl = 0;

void foo5() {
    ++gl;
}
