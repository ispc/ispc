#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        if (a == b) continue;
        ++a;
    }
    RET[programIndex] = a;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 6 + programIndex;
    RET[0] = RET[1] = RET[2] = RET[3] = RET[4] = 5;
}
