#include "test_static.isph"
task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double<3> a = { (double)aFOO[programIndex], (double)aFOO[programCount / 2 + programIndex], (double)aFOO[programCount * 2 + programIndex] };
    double<3> b = { (double)aFOO[programCount * 2 + programIndex], (double)aFOO[programCount - programIndex], (double)aFOO[programIndex] };
    double<3> c = max<double, 3>(a, b);
    RET[programIndex] = c[0];
    RET[programCount + programIndex] = c[1];
    RET[programCount * 2 + programIndex] = c[2];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programCount * 2 + programIndex;
    RET[programCount + programIndex] = max(1 + programCount / 2 + programIndex, 1 + programCount - programIndex);
    RET[programCount * 2 + programIndex] = 1 + programIndex;
}

