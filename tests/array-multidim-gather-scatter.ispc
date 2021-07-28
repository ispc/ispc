#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float x[40][40];
    for (uniform int i = 0; i < 40; ++i)
        for (uniform int j = 0; j < 40; ++j)
            x[i][j] = b;

    uniform int index[4] = { 0, 1, 2, 4 };
    #pragma ignore warning(perf)
    float v = index[programIndex & 0x3];
    #pragma ignore warning(perf)
    x[min(a,39.f)][v] = 0;
    #pragma ignore warning(perf)
    RET[programIndex] = x[v+1][v];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 5;
    RET[0] = RET[1] = RET[2] = 0;
}
