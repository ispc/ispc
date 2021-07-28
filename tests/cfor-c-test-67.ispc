#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        if (a == b) continue;
        ++a;
    }
    RET[programIndex] = a;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 5;
}
