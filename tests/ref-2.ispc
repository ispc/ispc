#include "../test_static.isph"
float foo(float & a) {
    a = 0;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    if (b > 6)
        foo(a);
    RET[programIndex] = a;
}

task void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
