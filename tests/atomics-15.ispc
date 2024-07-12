// rule: skip on arch=xe64
#include "../test_static.isph"
uniform float s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float delta = 1;
    #pragma ignore warning(perf)
    float b = atomic_add_global(&s, delta);
    RET[programIndex] = reduce_add(b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = reduce_add(programIndex);
}
