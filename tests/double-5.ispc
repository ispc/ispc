#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_di(uniform float RET[], uniform double aFOO[], uniform int bFOO[]) {
    double a = aFOO[programIndex];
    int b = bFOO[programIndex];
    if (a+4 == b)
        { RET[programIndex] = 1; return; }
    RET[programIndex] = 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
