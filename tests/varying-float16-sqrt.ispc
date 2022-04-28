#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    varying float16 f = programIndex/4;
    varying float16 calc = sqrt(f) * sqrt(f) - programIndex/4;
    if (calc < 0.0f16) calc = -calc;
    if (calc < 1e-2) {
        RET[programIndex] = 1;
    }
    else {
        RET[programIndex] = 0;
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
