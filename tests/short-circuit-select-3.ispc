#include "../test_static.isph"
float crashEven(uniform float a[]) {
    int offset = 0;
    return (programIndex & 1) ? a[offset] : a[offset+1000000];
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float a0 = aFOO[0], a1 = aFOO[1];
    if (((programIndex & 1) == 1) && (crashEven(aFOO) == 1))
        RET[programIndex] = 1;
    else
        RET[programIndex] = 2;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 1 : 2;
}
