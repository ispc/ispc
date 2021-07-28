#include "../test_static.isph"
uniform float a[100];



task void f_f(uniform float RET[], uniform float aFOO[]) {
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = i;
    RET[programIndex] = a[programIndex]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
