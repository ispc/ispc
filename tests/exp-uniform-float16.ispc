#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
uniform bool ok(uniform float16 x, uniform float16 ref) {
    return (abs(x - ref) < 1e-2f16) || abs((x - ref) / ref) < 1e-1f16;
}
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    for (uniform int i = 0; i != programCount; ++i) {
        uniform float16 ref = exp((uniform float16)((aFOO[i] / programCount) * 2));
        uniform float res = exp((aFOO[i] / programCount) * 2);
        RET[i] = ok(res, ref) ? 0. : 1.;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0.0; }
