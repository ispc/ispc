#include "../test_static.isph"
// rule: skip on arch=xe64

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32l*32l];
    for (uniform int i = 0; i < 32l*32l; ++i)
        buf[i] = i;

    #pragma ignore warning(perf)
    float a = buf[4*programIndex];
    #pragma ignore warning(perf)
    float b = buf[4*programIndex+1];
    #pragma ignore warning(perf)
    float c = buf[4*programIndex+2];
    #pragma ignore warning(perf)
    float d = buf[4*programIndex+3];

    RET[programIndex] = a+b+c+d;;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 4 * programIndex + 4 * programIndex + 1 + 
        4 * programIndex + 2 + 4 * programIndex + 3;
}
