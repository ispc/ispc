#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[4][3];
    uniform int i, j;
    for (i = 0; i < 4; ++i)
        for (j = 0; j < 3; ++j)
            x[i][j] = 3*i+j;

    a = clamp(a-2., 0., 2.);
    #pragma ignore warning(perf)
    RET[programIndex] = x[a][a];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 8;
    RET[0] = RET[1] = 0;
    RET[2] = 4;
}
