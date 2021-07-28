#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a = 1;
    uniform int * uniform b = &a;
    RET[programIndex] = *b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
