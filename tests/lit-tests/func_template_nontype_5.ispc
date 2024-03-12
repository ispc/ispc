// Basic test for templates with enum template parameters.
// RUN: %{ispc}  %s --emit-llvm-text --target=host --nostdlib -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} @enum_foo___uniCunenum_5B_Foo_5D_0___uni
// CHECK: ret i32 %k

// CHECK-LABEL: define {{.*}} @bar_uniform___uni
// CHECK: call i32 @enum_foo___uniCunenum_5B_Foo_5D_2___uni

// CHECK-LABEL: define linkonce_odr i32 @enum_foo___uniCunenum_5B_Foo_5D_2___uni
// CHECK: ret i32 2

// CHECK-LABEL: define {{.*}} @bar_varying___uni
// CHECK: call <{{[0-9]*}} x i32> @enum_foo___vyiCunenum_5B_Foo_5D_2___uni

// CHECK-LABEL: define linkonce_odr <{{[0-9]*}} x i32> @enum_foo___vyiCunenum_5B_Foo_5D_2___uni
// CHECK: ret {{.*}} i32 2

enum Foo {
    ZERO,
    ONE,
    TWO
};

template<typename T, Foo N>
noinline T enum_foo(uniform int k) {
    switch (N) {
        case ZERO:
            return 0;
        case ONE:
            return 1;
        case TWO:
            return 2;
    }
    return k;
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