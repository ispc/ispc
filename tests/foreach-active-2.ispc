#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex]; 
    uniform int count = 0;
    if (programIndex & 1)
        foreach_active (i)
            ++a;
    RET[programIndex] = a; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (1 + programIndex) + ((programIndex & 1) ? 1 : 0);
}
