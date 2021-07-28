#include "../test_static.isph"
void foo(float<3> &a) {
    a.x = 0;
    a.y *= 2;
    a.z = 1;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float<3> v = { a, b, 10 };
    if (a >= 2)
        foo(v);
    RET[programIndex] = v.x + v.y + v.z;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 11;
    RET[0] = 16;
}
