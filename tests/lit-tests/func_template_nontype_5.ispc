// Basic test for templates with enum template parameters.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: @enum_foo___uni0___uni
// CHECK: ret i32 %k

// CHECK-LABEL: @bar_uniform___uni
// CHECK: call i32 @enum_foo___uni2___uni

// CHECK-LABEL: linkonce_odr i32 @enum_foo___uni2___uni

// CHECK-LABEL: @bar_varying___uni
// CHECK: call <{{[0-9]*}} x i32> @enum_foo___vyi2___uni

// CHECK-LABEL: linkonce_odr <8 x i32> @enum_foo___vyi2___uni

enum Foo {
    ZERO,
    ONE,
    TWO
};

template<typename T, Foo N>
noinline T enum_foo(uniform int k) {
    return k + N;
}

template<>
noinline uniform int enum_foo<uniform int, ZERO>(uniform int k) {
    return k;
}

uniform int bar_uniform(uniform int k) {
    return enum_foo<uniform int, TWO>(k);
}

int bar_varying(uniform int k) {
    return enum_foo<varying int, TWO>(k);
}