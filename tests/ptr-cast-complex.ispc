#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int x[2][10];    
    for (uniform int i = 0; i < 2; ++i) {
        for (uniform int j = 0; j < 10; ++j) {
            x[i][j] = 10*i+j;
        }
    }

    uniform int (* varying y)[10] = x;
    #pragma ignore warning(perf)
    RET[programIndex] = y[1][programIndex % 5];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10+ (programIndex % 5);
}
