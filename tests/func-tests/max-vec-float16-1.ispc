#include "test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform float16<3> a = { (float16)aFOO[0], (float16)aFOO[4], (float16)aFOO[2] }; // 1 5 3
    uniform float16<3> b = { (float16)aFOO[1], (float16)aFOO[3], (float16)aFOO[0] }; // 2 4 1
    uniform float16<3> c = max<float16, 3>(a, b); // 2 5 3
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
