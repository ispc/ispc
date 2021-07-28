#include "../test_static.isph"
struct Foo {
    float x, y;
};

struct Bar {
    uniform Foo uf;
    varying Foo vf;
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Bar bar = { { b, b }, { a, a } };
    RET[programIndex] = bar.uf.x + bar.vf.y;
}

task void result(uniform float RET[]) { RET[programIndex] = 6+programIndex; }
