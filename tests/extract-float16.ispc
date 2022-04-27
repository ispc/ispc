#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float16 a = programIndex;
    RET[programIndex] = extract(a, min(programCount-1, 3));
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 0 : 3;
}
