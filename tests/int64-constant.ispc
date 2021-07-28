#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 v = 0xffffffff;
    RET[programIndex] = (v == 0xffffffffll) ? 1 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
