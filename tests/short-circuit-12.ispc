#include "../test_static.isph"


uniform int * uniform ptr;

float foo(uniform float a[]) {
    int index = (programIndex & 1) * 10000;
    #pragma ignore warning(perf)
    if (a[programIndex] < 10000 && a[index] == 1)
        return 1;
    else
        return 1234;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    float a0 = aFOO[0], a1 = aFOO[1];
    if ((programIndex & 1) == 0)
        RET[programIndex] = foo(aFOO);
    else
        RET[programIndex] = 2;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 2 : 1;
}
