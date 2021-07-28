#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a = 1;
    uniform int * uniform b = 0;
    RET[programIndex] = (b == NULL && b == 0 && 0 == b) ? 1 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
