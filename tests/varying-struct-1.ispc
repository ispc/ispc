#include "../test_static.isph"

struct Foo {
    uniform float x, y;
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo f[3] = { { b, b }, { 2*b, 2*b }, { 3*b, 3*b } };
    varying Foo g; 
    g = f[0];
    RET[programIndex] = g.x;
}

task void result(uniform float RET[]) { RET[programIndex] = 5; }
