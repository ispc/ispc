// RUN: not %{ispc} --target=host --nowrap --nostdlib %s -o - 2>&1 | FileCheck %s

// CHECK: Error: "void" type illegal for struct member

struct Foo {
    void x;
};

// CHECK: Error: "void" type global variable is illegal

void x;

// CHECK: Error: "void" type variable illegal in declaration

int func1() {
    void x;
}

// CHECK: Error: Parameter with type "void" illegal in function parameter list

void func2(void x, void y) {
    return x+y;
}