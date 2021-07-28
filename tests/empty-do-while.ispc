#include "../test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    do {} while (false);
    RET[programIndex] = (float)programIndex;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (float)programIndex;
}
