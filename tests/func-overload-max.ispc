#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = 1. / aFOO[programIndex]; 
    RET[programIndex] = max(0.f, a); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1. / (1+programIndex);
}
