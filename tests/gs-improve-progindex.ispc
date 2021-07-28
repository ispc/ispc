#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float udx[programCount+1];
    for (uniform int i = 0; i < programCount + 1; ++i)
        udx[i] = i+1;
    int x = programIndex;
    RET[programIndex] = udx[x];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex + 1;
}
