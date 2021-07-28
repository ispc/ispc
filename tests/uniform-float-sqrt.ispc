#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float f = aFOO[programCount/2];
    uniform float calc = sqrt(f) * sqrt(f) - aFOO[programCount/2];
    if (calc < 0.) calc = -calc;
    RET[programIndex] = (calc < 1e-5) ? 1 : 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
