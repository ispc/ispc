#include "../test_static.isph"
// rule: skip on cpu=TGLLP

task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    RET[programIndex] = a + b;
}

task void result(uniform float RET[]) { RET[programIndex] = 6 + programIndex; }
