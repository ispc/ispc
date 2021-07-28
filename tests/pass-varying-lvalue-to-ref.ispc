#include "../test_static.isph"
#pragma ignore warning(perf)
void inc(uniform float * varying v) { ++(*v); }

task void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    uniform float foo[2*programCount];
    for (uniform int i = 0; i < 2*programCount; ++i)
        foo[i] = 10+i;
    int a = (int)aa[programIndex];
    inc(&foo[a]);
    ret[programIndex] = foo[programIndex]-programIndex;
}

task void result(uniform float ret[]) {
    ret[programIndex] = 11;
    ret[0] = 10;
}
