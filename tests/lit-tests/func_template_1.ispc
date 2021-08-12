// Basic test to instantiate template with different atomic types as well as
// variability. Also checks case where an already instantiated instance is
// called.

// RUN: %{ispc}  %s --emit-llvm-text --target=host -o - | FileCheck %s

// CHECK: define <{{[0-9]*}} x float> @foo0
// CHECK: call <{{[0-9]*}} x i32> @goovyfvyi___vyfuni(<{{[0-9]*}} x float>
// CHECK: call <{{[0-9]*}} x i32> @goovyivyi___vyiuni(<{{[0-9]*}} x i32>

// CHECK: define <{{[0-9]*}} x float> @foo1___uniuni
// CHECK: call <{{[0-9]*}} x i32> @goovyivyi___vyiuni(<{{[0-9]*}} x i32>
// CHECK: call <{{[0-9]*}} x i32> @goounivyi___uniuni(i32

// CHECK: define <{{[0-9]*}} x float> @foo2___uniuni
// CHECK: call <{{[0-9]*}} x i32> @goovyivyi___vyiuni(<{{[0-9]*}} x i32>
// CHECK: call <{{[0-9]*}} x i32> @goovyfvyi___vyfuni(<{{[0-9]*}} x float>

template <typename T, typename C> noinline int goo(T argGooOne, uniform int argGooTwo) {
    C val = argGooTwo;
    int i_val = argGooOne;
    return val + i_val;
}

float foo0(uniform int argFoo0, uniform int argFoo1) {
    int a = goo<float, int>(argFoo0, argFoo1);
    int b = goo<int, int>(argFoo0, argFoo1);
    return a + b;
}

float foo1(uniform int argFoo0, uniform int argFoo1) {
    int a = goo<int, int>(argFoo0, argFoo1);
    int b = goo<uniform int, int>(argFoo0, argFoo1);
    return a + b;
}

float foo2(uniform int argFoo0, uniform int argFoo1) {
    int a = goo<int, int>(argFoo0, argFoo1);
    int b = goo<float, int>(argFoo0, argFoo1);
    return a + b;
}
