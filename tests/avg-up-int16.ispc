#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float bf) {
    int16 a = aFOO[programIndex];
    int16 b = bf;
    RET[programIndex] = avg_up(a, b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = ((int)programIndex + 1 + 5 + 1) / 2;
}
