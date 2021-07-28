#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    int a = 10 + aFOO[programIndex/2];
    RET[programIndex] = 1;
    uniform int sameVal;
    uniform bool re;
    if (a <= 11) {
        re = reduce_equal(a, &sameVal);
//CO        print("% % %\n", re, sameVal, a);
    }
    RET[programIndex] = ((int)re << 8) + sameVal;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 267;
}
