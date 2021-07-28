#include "../test_static.isph"
float f() { return 2.; }
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = 1. + f() + 2.; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
