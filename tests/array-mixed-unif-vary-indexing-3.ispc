#include "../test_static.isph"
// constant 6 here is random small positive number. It needs to be >=1.
#define SIZE TARGET_WIDTH+6

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[SIZE][SIZE];
    for (uniform int i = 0; i < SIZE; ++i)
        for (uniform int j = 0; j < SIZE; ++j)
            x[i][j] = 2+b-5;

    // all are 2 except (4,2) = 0, (4,...) = 1, (4,programCount-1)=2
    if (a == 3.) {
        #pragma ignore warning(perf)
        x[b-1][a-1] = 0;
    }
    else {
        #pragma ignore warning(perf)
        x[b-1][a-1] = 1;
    }
    #pragma ignore warning(perf)
    RET[programIndex] = x[4][a];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
    RET[1] = 0;
    RET[programCount-1] = 2;
}
