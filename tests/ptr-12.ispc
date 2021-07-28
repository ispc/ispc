#include "../test_static.isph"
void inc(int * uniform v) {
    ++*v;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * uniform b = &aFOO[0];
    ++b;
    b++;
    RET[programIndex] = *b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3;
}
