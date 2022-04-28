#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float16 a = aFOO[programIndex&1];
    RET[programIndex] = 1;
    uniform bool re = false;
    uniform float16 val = 16.0f16;
    if (programIndex & 1) {
        re = reduce_equal(a, &val);
    }
    RET[programIndex] = ((int)re << 8) + val;
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 16 : 258;
}
