#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float<3> a = { aFOO[programIndex], aFOO[programCount / 2 + programIndex], aFOO[programCount * 2 + programIndex] };
    float<3> b = { aFOO[programCount * 2 + programIndex], aFOO[programCount - programIndex], aFOO[programIndex] };
    float<3> c = max<float, 3>(a, b);
    RET[programIndex] = c[0];
    RET[programCount + programIndex] = c[1];
    RET[programCount * 2 + programIndex] = c[2];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programCount * 2 + programIndex;
    RET[programCount + programIndex] = max(1 + programCount / 2 + programIndex, 1 + programCount - programIndex);
    RET[programCount * 2 + programIndex] = 1 + programIndex;
}
