#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float i, r = 0.;
    cfor (i = 0; i < a+b; ++i)
        r += b;
    RET[programIndex] = r;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 30 + 5 * programIndex;
}
