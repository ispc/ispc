#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = sizeof(uniform double); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 8;
}
