#include "../test_static.isph"
uniform unsigned int32 s = 0;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    if (programIndex & 1) {
        #pragma ignore warning(perf)
        b = atomic_add_global(&s, programIndex);
    }
    RET[programIndex] = s;
}

task void result(uniform float RET[]) {
    uniform int sum = 0;
    for (uniform int i = 0; i < programCount; ++i)
        if (i & 1)
            sum += i;
    RET[programIndex] = sum;
}
