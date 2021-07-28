#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[programCount+5];
    for (uniform int i = 0; i < programCount+5; ++i)
        x[i] = 0.;
    #pragma ignore warning(perf)
    x[a] = 2;
    RET[programIndex] = x[4] + x[0] + x[5];
}

    
task void result(uniform float RET[]) {
    if (programCount == 1)
        RET[programIndex] = 0;
    else if (programCount == 4)
        RET[programIndex] = 2;
    else
        RET[programIndex] = 4;
}
