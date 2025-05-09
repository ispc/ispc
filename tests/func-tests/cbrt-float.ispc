#include "test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    varying float f = aFOO[programIndex];
    varying float calc = cbrt(f) * cbrt(f) * cbrt(f);
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
