#include "test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    varying float f = aFOO[programIndex];
    varying float calc = cbrt(f) * cbrt(f) * cbrt(f);
    if (isnan(calc) == isnan(f) && (abs(f) >= 1e-6 && abs((calc - f) / f) < 2e-6 || abs(calc - f) < 2e-12 || isnan(f))) {
        RET[programIndex] = 1;
    }
    else {
        RET[programIndex] = 0;
    }
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
