#include "../test_static.isph"
struct Foo {
    float x, y;
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    varying Foo g;
    g.x = a;
    RET[programIndex] = g.x;
}

task void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
