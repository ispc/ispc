#include "../test_static.isph"
// rule: skip on arch=xe32
// rule: skip on arch=xe64

static float array[6];
task void x(uniform int i, float f) {
    array[i] = f;
}
task void f_fu(uniform float RET[], uniform float fFOO[], uniform float fu) {
    float f = fFOO[programIndex];
    launch x(fu, f);
    sync;
    RET[programIndex] = array[5];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
