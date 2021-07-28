#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform int x;
    float aa = a;
    cfor (x = 0; x < 99999; ++x) {
        if (x == a) break;
        ++aa;
    }
    RET[programIndex] = aa;
}

task void result(uniform float RET[]) { RET[programIndex] = 2 + 2*programIndex; }
