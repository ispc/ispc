#include "../test_static.isph"
uniform unsigned int64 s = 0xffffffffff000000;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    float b = 0;
    if (programIndex & 1) {
        #pragma ignore warning(perf)
        b = atomic_or_global(&s, (1ull << programIndex));
    }
    RET[programIndex] = (s>>20);
}

task void result(uniform float RET[]) {
    uniform int64 sum = 0;
    for (uniform int i = 0; i < programCount; ++i)
        if (i & 1)
            sum += (1ull << i);
    RET[programIndex] = ((unsigned int64)(0xffffffffff000000 | sum)) >> 20;
}
