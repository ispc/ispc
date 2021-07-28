#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[2*programCount];
    for (uniform int i = 0; i < 2*programCount; ++i)
        x[i] = 0;

    #pragma ignore warning(perf)
    x[2*(a-1)] = b;
    #pragma ignore warning(perf)
    RET[programIndex] = x[2*(a-1)];
}

    
task void result(uniform float RET[]) { RET[programIndex] = 5; }
