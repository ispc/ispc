#include "test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    uniform double<3> a = { aFOO[0], aFOO[4], aFOO[2] };
    uniform double<3> b = { aFOO[1], aFOO[3], aFOO[0] };
    uniform double<3> c = max<double, 3>(a, b);
    RET[programIndex] = 0;
    RET[0] = c[0];
    RET[1] = c[1];
    RET[2] = c[2];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 0;
    RET[0] = 2;
    RET[1] = 5;
    RET[2] = 3;
}

