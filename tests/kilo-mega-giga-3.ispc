#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    unsigned int32 a = 3G;
    a -= 2G;
    a -= 1024M;
    #pragma ignore warning(perf)
    RET[programIndex] = a;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
