#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[programCount+5];
    uniform int i;
    for (i = 0; i < programCount+5; ++i)
        x[i] = i+b;
    a -= 1;
    if (a == 3) a = 0;
    a *= 10000000;
    float ret = 1234;
    if (a < 5) {
        #pragma ignore warning(perf)
        ret = x[a];
    }
    RET[programIndex] = ret;
}


task void result(uniform float RET[4]) {
    RET[programIndex] = 1234;
    RET[0] = RET[3] = 5;
}
