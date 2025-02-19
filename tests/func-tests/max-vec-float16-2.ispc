#include "test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16<3> a = { (float16)aFOO[programIndex], (float16)aFOO[programCount / 2 + programIndex], (float16)aFOO[programCount * 2 + programIndex] };
    float16<3> b = { (float16)aFOO[programCount * 2 + programIndex], (float16)aFOO[programCount - programIndex], (float16)aFOO[programIndex] };
    float16<3> c = max<float16, 3>(a, b);
    RET[programIndex] = c[0];
    RET[programCount + programIndex] = c[1];
    RET[programCount * 2 + programIndex] = c[2];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programCount * 2 + programIndex;
    RET[programCount + programIndex] = max(1 + programCount / 2 + programIndex, 1 + programCount - programIndex);
    RET[programCount * 2 + programIndex] = 1 + programIndex;
}

