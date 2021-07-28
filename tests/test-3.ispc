#include "../test_static.isph"
float foo(float x) {
    return x + x; 
}


task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b; 
    b = foo(a); 
    RET[programIndex] = b; 
}


task void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
