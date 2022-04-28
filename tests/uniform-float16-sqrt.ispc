#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float16 f = aFOO[programCount]/4;
    uniform float16 calc = sqrt(f) * sqrt(f) - aFOO[programCount]/4;
    if (calc < 0.0f16) calc = -calc;
    RET[programIndex] = (calc < 1e-2) ? 1 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
