#include "../test_static.isph"
bool bar(float * x) {
    return (x != 0);
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = bar(NULL);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
