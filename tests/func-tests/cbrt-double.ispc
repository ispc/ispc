#include "test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2
task void f_di(uniform float RET[], uniform double aFOO[], uniform int b[]) {
    varying double f = aFOO[programIndex];
    varying double calc = cbrt(f) * cbrt(f) * cbrt(f);
    if (calc < 0.) calc = -calc;
    if (abs(calc - f) < 1e-5)
    {
        RET[programIndex] = 1;
    }
    else
    {
        RET[programIndex] = 0;
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
