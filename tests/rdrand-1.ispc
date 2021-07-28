#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
#ifndef ISPC_HAS_RAND
    RET[programIndex] = 1;
#else

    uniform float r = -1;
    uniform int count = 0;
    while (!rdrand(&r)) {
        ++count;
    }
    RET[programIndex] = (r >= 0 && r < 1);

#endif
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
