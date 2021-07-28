#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float i;
    #pragma ignore warning
    cfor (i = 0; i < b; ++i) {
        if (i+2 == b) continue;
        ++a;
    }
    RET[programIndex] = a;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 5 + programIndex;
}
