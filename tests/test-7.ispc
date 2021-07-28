#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    RET[programIndex] = a + 2. * b;
}


task void result(uniform float RET[4]) {
    RET[programIndex] = 11 + programIndex;
}
