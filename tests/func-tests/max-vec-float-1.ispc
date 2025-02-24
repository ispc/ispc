#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform float<3> a = { aFOO[0], aFOO[4], aFOO[2] }; // 1 5 3
    uniform float<3> b = { aFOO[1], aFOO[3], aFOO[0] }; // 2 4 1
    uniform float<3> c = max<float, 3>(a, b); // 2 5 3
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
