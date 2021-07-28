#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = round(a+.49999); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
