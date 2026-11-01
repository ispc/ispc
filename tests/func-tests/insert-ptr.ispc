#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * varying p = &aFOO[0];
    for (uniform int i = 0; i < programCount; i++)
        p = insert(p, i, &aFOO[programCount - 1 - i]);
    RET[programIndex] = *p;
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount - programIndex;
}
