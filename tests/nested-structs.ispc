#include "../test_static.isph"
struct Foo {
    float f[6];
};

struct Bar {
    uniform Foo foo[6];
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform Bar bar;
    for (uniform int i = 0; i < 6; ++i)
        for (uniform int j = 0; j < 6; ++j)
            bar.foo[i].f[j] = 2.+b-5;
    RET[programIndex] = bar.foo[b].f[b];
}

task void result(uniform float RET[]) { RET[programIndex] = 2; }
