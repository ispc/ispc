#include "../test_static.isph"
// constant 5 here is random small positive number. It needs to be >=1.
#define SIZE TARGET_WIDTH+5

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[SIZE][SIZE];
    for (uniform int i = 0; i < SIZE; ++i)
        for (uniform int j = 0; j < SIZE; ++j)
            x[i][j] = 2+b-5;

    // all are 2 except (3,4) = 0, (1,4) = 1, (2,4) = 1, (4,4) = 1
    if (a == 3.) {
        #pragma ignore warning(perf)
        x[a][b-1] = 0;
    }
    else {
        #pragma ignore warning(perf)
        x[a][b-1] = 1;
    }

    a = min(a, (float)SIZE-1);

    #pragma ignore warning(perf)
    RET[programIndex] = x[3][a];
}

task void result(uniform float RET[]) { RET[programIndex] = 2; RET[3] = 0; }
