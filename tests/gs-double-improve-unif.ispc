#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[4] = { 1,2,3,4 };
    int x = b - 5;
    RET[programIndex] = udx[x];
}

task void result(uniform float RET[]) { RET[programIndex] = 1; }
