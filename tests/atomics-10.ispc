#include "../test_static.isph"
uniform unsigned int32 s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    float delta = 1;
    if (programIndex < 2) {
        #pragma ignore warning(perf)
        b = atomic_add_global(&s, delta);
    }
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    RET[programIndex] = programCount == 1 ? 1 : 2;
}
