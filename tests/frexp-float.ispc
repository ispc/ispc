#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = (1ul << (programIndex%28)) * 1.5;
    if (programIndex & 1)
        a = -a;
    int exponent;
    RET[programIndex] = frexp(a, &exponent);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0.75;
    if (programIndex & 1)
        RET[programIndex] = -RET[programIndex];
}
