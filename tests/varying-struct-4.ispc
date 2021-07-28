#include "../test_static.isph"
struct Foo {
    float x, y;
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Foo f[3] = { { b, 2*b }, { 3*b, 4*b }, { 5*b, 6*b } };
    int index = (a <= 2) ? 1 : 2;
    #pragma ignore warning(perf)
    varying Foo g = f[index];
    RET[programIndex] = g.x;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 25;
    RET[0] = RET[1] = 15;
}
