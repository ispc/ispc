#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 a = aFOO[programIndex];
    uniform float16 a0 = RET[0];
    RET[programIndex] = select(programIndex % 2, a, (float16)b);
    RET[0] = select(b > 3, max(a0, b), 0.0f16);
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex % 2) ? programIndex + 1 : 5;
    RET[0] = 5;
}
