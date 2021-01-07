#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_duf(uniform float RET[], uniform double aFOO[], uniform float b) {
    double a = aFOO[programIndex];
    RET[programIndex] = a / b;
}

task void result(uniform float RET[]) { RET[programIndex] = (1+programIndex)/5.; }
