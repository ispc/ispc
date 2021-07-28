#include "../test_static.isph"
noinline void swap(bool doSwap, float &a_foo, float &b_foo) { a_foo = doSwap ? b_foo : a_foo; }

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a_foo = -1;
    float b_foo = aFOO[programIndex];
    RET[programIndex] = 0;

    bool doswap = true;
    swap(doswap, a_foo, b_foo);

    if (a_foo != b_foo)
        RET[programIndex] = 1;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
