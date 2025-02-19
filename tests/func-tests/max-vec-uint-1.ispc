#include "test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform uint<3> a = { (uint)aFOO[0], (uint)aFOO[4], (uint)aFOO[2] }; // 1 5 3
    uniform uint<3> b = { (uint)aFOO[1], (uint)aFOO[3], (uint)aFOO[0] }; // 2 4 1
    uniform uint<3> c = max<uint, 3>(a, b); // 2 5 3
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

