#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float16 aa = aFOO[programIndex];
    float16 bb = aa + programCount;
    float16 shuf = shuffle(aa, bb, 2 * programIndex + (int)b - 5);
    RET[programIndex] = shuf;
}

task void result(uniform float RET[]) { RET[programIndex] = 1 + 2 * programIndex; }
