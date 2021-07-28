#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[3][5] = { { 0, 1, 2, 3, 4 }, { 5, 6, 7, 8, 9 }, 
                              { 10, 11, 12, 13, 14 } };
    RET[programIndex] = a * x[1][b-4];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 6 * (1+programIndex);
}
