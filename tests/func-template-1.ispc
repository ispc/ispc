#include "../test_static.isph"

template <typename T, typename C> noinline uniform float noinline goo(T argGooOne, C argGooTwo) { return 0; }

template <> noinline uniform float noinline goo<int, int>(int argGooOne, int argGooTwo) { return 1; }

template <> noinline uniform float noinline goo<uniform float, int>(uniform float argGooOne, int argGooTwo) {
    return argGooOne;
}

template <typename C> uniform float noinline goo(C argGooOne) { return 2; }

template <> uniform float noinline goo<float>(float argGooOne) { return 3; }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b_val) {
    int argFoo0 = aFOO[programIndex];
    int argFoo1 = aFOO[programIndex] / 2;
    RET[programIndex] = goo<int, uniform float>(argFoo0, b_val); // primary template, return 0
    RET[0] = goo<float>((float)argFoo0);                         // specialized template, return 3
    RET[1] = goo<int>(argFoo0, argFoo1);                         // primary template with two args, return 1
    RET[2] = goo<int, int>(argFoo0, argFoo1);                    // specialized template, return 1
    RET[3] = goo<uniform float, int>(b_val, argFoo1);            // specialized template, return b_val
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 3;
    RET[1] = 1;
    RET[2] = 1;
    RET[3] = 5;
}
