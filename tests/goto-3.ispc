#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    float b = 0.; b = a; 
    RET[programIndex] = a+b; 
    if (all(a == 0))
        goto skip;
    RET[programIndex] = 0; 
 skip:
    ;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
