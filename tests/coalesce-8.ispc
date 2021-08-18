#include "../test_static.isph"
// rule: skip on arch=xe32
// rule: skip on arch=xe64

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform buf = uniform new uniform float[32l*32l];
    for (uniform int i = 0; i < 32l*32l; ++i)
        buf[i] = i;

    int index = (programIndex < 4) ? (programIndex & 1) :
        (programIndex / 4);
    #pragma ignore warning(perf)
    float a = buf[index];

    RET[programIndex] = a;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex < 4) ? (programIndex & 1) :
        (programIndex / 4);
}
