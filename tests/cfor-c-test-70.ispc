#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    if (a >= 4)
        a = 0;
    else {
        cfor (i = 0; i < b; ++i) {
            if (a == b) 
                a += 10;
            ++a;
            if (a == 2) break;
        }
    }
    RET[programIndex] = a;
}

task void result(uniform float RET[]) {
    uniform float ret[4] = { 2,17,18,0 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
