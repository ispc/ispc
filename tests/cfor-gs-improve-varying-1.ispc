#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float udx[5];
    #pragma ignore warning
    cfor (uniform int i = 0; i < 5; ++i)
        udx[i] = i + a;

    int x = b-4;
    #pragma ignore warning(perf)
    RET[programIndex] = udx[x];
}

task void result(uniform float RET[]) { RET[programIndex] = 2+programIndex; }
