#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    int64 a = aFOO[programIndex];
    int64 b = -2. * a;
    RET[programIndex] = min(a,b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = -2 * (1 + programIndex);
}
