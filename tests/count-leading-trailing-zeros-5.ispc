#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    unsigned int64 i = ((unsigned int64)1 << min(63, 40+programIndex));
    #pragma ignore warning(perf) 
    RET[programIndex] = count_trailing_zeros(i);
}

task void result(uniform float RET[]) {
    RET[programIndex] = min(63, 40+programIndex);
}
