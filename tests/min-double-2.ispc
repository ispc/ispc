#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    RET[programIndex] = min(-10 * (a-3), (double).1f);
    RET[programCount-1] = min(-10 * b, (double)2);
}

task void result(uniform float RET[]) {
    RET[programIndex] = -10 * (programIndex - 2);
    RET[0] = .1;
    RET[1] = .1;
    RET[programCount - 1] = -50;
}
