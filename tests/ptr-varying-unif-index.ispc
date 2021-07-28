#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float buf[programCount];
    uniform float * varying ptr = &buf[programCount - 1 - programIndex];
    #pragma ignore warning(perf)
    ptr[0] = programIndex;
    RET[programIndex] = buf[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount - 1 - programIndex;
}
