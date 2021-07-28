#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    RET[programIndex] = reduce_equal(a); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 1 : 0;
}
