// rule: skip on arch=ppc64le
#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float16 a = aFOO[programIndex];
    float16 b = (programCount == 1) ? 4 : broadcast(a, 2);
    RET[programIndex] = b;
}

task void result(uniform float RET[]) { RET[programIndex] = 3; }
