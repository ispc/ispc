#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float udx[4] = { 1,2,3,4 };
    int x = b - 5;
    RET[programIndex] = udx[x];
}

task void result(uniform float RET[]) { RET[programIndex] = 1; }
