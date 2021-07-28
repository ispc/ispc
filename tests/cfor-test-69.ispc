#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        if (a == b) 
            a += 10;
        ++a;
        if (a == 2) break;
    }
    RET[programIndex] = a;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 6 + programIndex;
    RET[0] = 2;
    RET[1] = 17;
    RET[2] = 18;
    RET[3] = 19;
    RET[4] = 20;
}
