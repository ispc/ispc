#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    #pragma ignore warning(perf)
    unsigned int64 a = aFOO[programIndex];
    unsigned int64 b = 2. * a;
    #pragma ignore warning(perf)
    RET[programIndex] = min(a,b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
