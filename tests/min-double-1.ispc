#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    RET[programIndex] = min(3 * a, (double)10.f);
    RET[programCount-1] = min(b, (double)100);
}


task void result(uniform float RET[]) {
    RET[programIndex] = 10;
    RET[0] = 3;
    RET[1] = 6;
    RET[2] = 9;
    RET[programCount-1] = 5;
}
