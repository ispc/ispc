#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float16 a = (1ul << ((programIndex/4)%28)) * 1.5;
    if (programIndex & 1)
        a = -a;
    int exponent;
    frexp(a, &exponent);
    RET[programIndex] = exponent;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+((programIndex/4)%28);
}
