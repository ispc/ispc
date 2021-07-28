#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform ptr = aFOO;
    RET[programIndex] = *(ptr + programIndex) - 1;
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
