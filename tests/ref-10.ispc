#include "../test_static.isph"
struct Foo {
    float a, b, c;
};

void foo(Foo &f, uniform int dim) {
    (&f.a)[dim] += (&f.a)[dim];
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    Foo f = { b, aFOO[programIndex], 1 };
    if (programIndex < 2)
        foo(f, 0);
    RET[programIndex] = f.a + f.b + f.c;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 7+programIndex;
    RET[0] = 12;
    RET[1] = 13;
}
