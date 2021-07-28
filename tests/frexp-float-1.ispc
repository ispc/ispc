#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = (1ul << (programIndex%28)) * 1.5;
    if (programIndex & 1)
        a = -a;
    int exponent;
    frexp(a, &exponent);
    RET[programIndex] = exponent;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+(programIndex%28);
}
