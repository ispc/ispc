// RUN: %{ispc} --target=host --nowrap --nostdlib --emit-llvm-text %s -o - | FileCheck %s

enum E {
    E1 = 1,
    E2 = 2,
    E3 = 4,
};
static const uniform unsigned int e = E1 | E3;

// CHECK-LABEL: @foo
// CHECK-NEXT: allocas:
// CHECK-NEXT: ret i32 5
uniform int foo() {
    return e;
}

// CHECK-LABEL: @bar
// CHECK-NEXT: allocas:
// CHECK-NEXT: ret i32 3
uniform int bar() {
    static const uniform unsigned int x = E1 | 2;
    return x;
}

// CHECK-LABEL: @func
// CHECK-NEXT: allocas:
// CHECK-NEXT: ret i32 6
uniform int func() {
    static const uniform unsigned int x = 4 | E2;
    return x;
}

// CHECK-LABEL: @boo
// CHECK-NEXT: allocas:
// CHECK-NEXT: ret i32 7
uniform int boo() {
    static const uniform unsigned int x = ((E)4 ^ E1) ^ E2;
    return x;
}
