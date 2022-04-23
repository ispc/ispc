#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 a = aFOO[programIndex];
    RET[programIndex] = max(3 * a, 10.f16);
    RET[programCount - 1] = max((float16)b, 100.f16);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 3 * (1 + programIndex);
    RET[0] = 10;
    RET[1] = 10;
    RET[2] = 10;
    RET[programCount - 1] = 100;
}
