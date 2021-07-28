#include "../test_static.isph"
//add test for f(int x[2][10]) - f(int x[][10]) - f(int (*x)[10])x



int foo(uniform int x[2][10]) {
    #pragma ignore warning(perf)
    return x[1][programIndex % 5];
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int x[2][10];    
    for (uniform int i = 0; i < 2; ++i) {
        for (uniform int j = 0; j < 10; ++j) {
            x[i][j] = 10*i+j;
        }
    }
    
    RET[programIndex] = foo(x);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10+ (programIndex % 5);
}
