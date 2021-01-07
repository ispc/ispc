#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    double v = aFOO[programIndex];
    uniform float m;
    if (v > 0 && v < 3)
        m = reduce_min(-v);
    RET[programIndex] = m;
}

task void result(uniform float RET[]) { RET[programIndex] = (programCount == 1) ? -1 : -2; }
