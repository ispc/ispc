#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
#ifndef ISPC_HAS_RAND
    RET[programIndex] = 1;
#else

    float r = -1;
    while (!rdrand(&r))
        ;
    RET[programIndex] = (r >= 0 && r < 1);

#endif
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
