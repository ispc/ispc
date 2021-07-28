#include "../test_static.isph"
// constant 5 here is random small positive number. It needs to be >=1.
#define SIZE TARGET_WIDTH+5

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[SIZE][SIZE];
    for (uniform int i = 0; i < SIZE; ++i)
        for (uniform int j = 0; j < SIZE; ++j)
            x[i][j] = 2+b-5;

    a = min(a, (float)SIZE-1);
    #pragma ignore warning(perf)
    x[a][b-1] = 0;
    #pragma ignore warning(perf)
    RET[programIndex] = x[2][a];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2;
    RET[3] = 0;
}
