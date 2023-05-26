#include "../test_static.isph"
// rule: skip on arch=xe64

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32l*32l];
    for (uniform int i = 0; i < 32l*32l; ++i)
        buf[i] = i;

    #pragma ignore warning(perf)
    RET[programIndex] = buf[programIndex & 1];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex & 1;
}
