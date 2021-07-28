#include "../test_static.isph"
void inc(int * uniform v) {
    ++*v;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = aFOO;
    b[programIndex] = programCount - programIndex;
    RET[programIndex] = aFOO[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount - programIndex;
}
