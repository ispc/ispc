#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    RET[programIndex] = max(-10 * (a-3), (double).1f);
    RET[programCount-1] = max(-10 * b, (double)2);
}

task void result(uniform float RET[]) {
    RET[programIndex] = .1;
    RET[0] = 20;
    RET[1] = 10;
    RET[programCount - 1] = 2;
}
