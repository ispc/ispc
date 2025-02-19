#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int<3> a = { (int)aFOO[0], (int)aFOO[4], (int)aFOO[2] }; // 1 5 3
    uniform int<3> b = { (int)aFOO[1], (int)aFOO[3], (int)aFOO[0] }; // 2 4 1
    uniform int<3> c = max<int, 3>(a, b); // 2 5 3
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
