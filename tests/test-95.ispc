#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform int i;
    float x[10];
    for (i = 0; i < 10; ++i)
        x[i] = i;
    RET[programIndex] = x[b];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 5;
}
