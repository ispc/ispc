#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    int<3> a = { (int)aFOO[programIndex], (int)aFOO[programCount / 2 + programIndex], (int)aFOO[programCount * 2 + programIndex] };
    int<3> b = { (int)aFOO[programCount * 2 + programIndex], (int)aFOO[programCount - programIndex], (int)aFOO[programIndex] };
    int<3> c = max<int, 3>(a, b);
    RET[programIndex] = c[0];
    RET[programCount + programIndex] = c[1];
    RET[programCount * 2 + programIndex] = c[2];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programCount * 2 + programIndex;
    RET[programCount + programIndex] = max(1 + programCount / 2 + programIndex, 1 + programCount - programIndex);
    RET[programCount * 2 + programIndex] = 1 + programIndex;
}

