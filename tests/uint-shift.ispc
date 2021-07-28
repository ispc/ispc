#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    unsigned int x = 0xffffffff;
    #pragma ignore warning(perf)
    RET[programIndex] = x >> 16; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0xffff;
}
