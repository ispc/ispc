#include "../test_static.isph"
uniform float a[1234];

float * uniform func(uniform bool x) {
    return x ? a : NULL;
}


task void f_f(uniform float RET[], uniform float aFOO[]) {
    a[programIndex] = aFOO[programIndex];
    float * uniform ptr = func(aFOO[0] == 1);
    RET[programIndex] = ptr[programIndex]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
