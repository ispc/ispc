#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int8 * varying pa = (uniform int8 *)aFOO;
    uniform int8 * varying pb = (uniform int8 *)(&aFOO[10]);
    RET[programIndex] = pb - pa;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 40;
}
