#include "../test_static.isph"
// rule: skip on arch=xe64

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    float * uniform buf = uniform new float[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        buf[i] = i;
    #pragma ignore warning(perf)
    RET[programIndex] = buf[a-1];
    delete buf;
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
