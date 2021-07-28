#include "../test_static.isph"
uniform int foo(int i);

task void f_f(uniform float RET[], uniform float aFOO[]) {
    for (uniform int i = 0; i < programCount; ++i)
        RET[i] = 0;

    foreach (i = 2 ... 0)
        RET[i] -= 1234;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
