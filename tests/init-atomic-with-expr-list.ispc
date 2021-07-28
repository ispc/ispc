#include "../test_static.isph"
int b = { 2. }; 

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float aa = { a };
    RET[programIndex] = aa+b; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3 + 1*programIndex;
}
