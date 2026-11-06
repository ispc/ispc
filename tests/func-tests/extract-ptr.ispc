#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float * varying p = &aFOO[programCount - 1 - programIndex];
    for (uniform int i = 0; i < programCount; i++) {
        uniform float * uniform up = extract(p, i);
        RET[i] = *up;
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount - programIndex;
}
