#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_duf(uniform float RET[], uniform double aFOO[], uniform float b) {
    double a = aFOO[programIndex];
    if (a+3 > b)
        { RET[programIndex] = 1; return; }
    RET[programIndex] = 0;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
    RET[0] = RET[1] = 0;
}
