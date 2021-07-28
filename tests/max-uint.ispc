#include "../test_static.isph"
task void f_f(uniform float result[], uniform float aa[]) {
    #pragma ignore warning(perf)
    unsigned int i = (unsigned int)aa[programIndex];
    #pragma ignore warning(perf)
    result[programIndex] = max((unsigned int)100, i);
}

task void result(uniform float r[]) { r[programIndex] = 100; }
