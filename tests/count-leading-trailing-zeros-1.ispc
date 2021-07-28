#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = count_trailing_zeros(0xf0ul);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
