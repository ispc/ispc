#include "../test_static.isph"


 int * uniform ptr;

int crash() {
    return *ptr;
}

float foo(uniform float a[]) {
    int index = (programIndex & 1);
    #pragma ignore warning(perf)
    if (a[index] == 2 && crash())
        return 1234;
    else
        return 1;
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
