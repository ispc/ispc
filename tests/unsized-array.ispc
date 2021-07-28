#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a[][2][] = { { { 1 }, { 2 } }, { { 3 }, { 4 } }, { { 5 }, { 6 } } };
    RET[programIndex] = a[1][1][0]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
