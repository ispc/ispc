#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, r = a;
    if (a >= 4)
        { RET[programIndex] = 0; return; }
    else {
        cfor (i = 0; i < a; ++i) {
            if (r == b) 
                r += 10;
            ++r;
            if (r == 2) break;
        }
    }
    RET[programIndex] = r;
}

task void result(uniform float RET[]) {
    uniform float ret[4] = { 2,4,16,0 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
