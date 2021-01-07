#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex];
    double b = 2. * a;
    RET[programIndex] = min(a,b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
