#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    if (false) RET[programIndex] = a; 
    else RET[programIndex] = 1230 + 4; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1234.000000;
}
