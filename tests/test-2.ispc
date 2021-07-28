#include "../test_static.isph"
float foo(uniform float xFOO[]) {
    float x = xFOO[programIndex];
    return x + x; 
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = foo(aFOO); 
}


task void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
