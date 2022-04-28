#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 a = aFOO[programIndex];
    RET[programIndex] = 0;
    if (programIndex & 1)
        RET[programIndex] = reduce_equal(a) ? 1 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
