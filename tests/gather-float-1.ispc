#include "../test_static.isph"
int zero = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float a[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = aFOO[i];

    #pragma ignore warning(perf)
    int g = a[programIndex+zero]; 
    RET[programIndex] = g; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
