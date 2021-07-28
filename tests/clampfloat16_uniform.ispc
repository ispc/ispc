#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = 10;
    uniform float16 a = -4.f16;
    RET[0] = clamp((uniform float16)a, -10.f16, -5.f16);
    RET[1] = clamp((uniform float16)2 * a, -10.f16, -5.f16);
    RET[2] = clamp((uniform float16)-3 * a, 5.f16, 10.f16);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10;
    RET[0] = -5;
    RET[1] = -8;
    RET[2] = 10;
}
