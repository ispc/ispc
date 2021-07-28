#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float i) {
    float a = aFOO[programIndex];
    uniform float x[6] = { 0, 1, 2, 3, 4, 2 };
    RET[programIndex] = a * x[i];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2 * programIndex;
}
