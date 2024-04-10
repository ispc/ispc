// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: Arrays of "void" type are illegal

uniform void aFOO[] = { NULL };

// CHECK: Error: Arrays of "void" type are illegal

struct Foo {
    void aFOO[];
};

// CHECK: Error: Arrays of "void" type are illegal

float f_fu(uniform void aFOO[]) {
    return 0;
}

// CHECK: Error: Illegal to dereference void pointer type 

int foo(void * uniform ptr) {
    print("%", ptr[0]);
}

// CHECK: Error: Illegal to dereference void pointer type

int func(void * uniform ptr) {
    print("%", *ptr);
}