#include "../test_static.isph"
task void f_f(uniform float r[], uniform float a[]) {
    #pragma ignore warning(perf)
    unsigned int i = (unsigned int)a[programIndex];
    #pragma ignore warning(perf)
    r[programIndex] =  min((unsigned int)200, i);
}

task void result(uniform float r[]) {
    r[programIndex] = 1+programIndex;
}
