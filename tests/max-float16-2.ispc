#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 a = aFOO[programIndex];
    RET[programIndex] = max(-10 * (a - 3), .5f16);
    RET[programCount - 1] = max(-10 * (float16)b, 2.f16);
}

task void result(uniform float RET[]) {
    RET[programIndex] = .5f16;
    RET[0] = 20;
    RET[1] = 10;
    RET[programCount - 1] = 2;
}
