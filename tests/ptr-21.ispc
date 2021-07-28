#include "../test_static.isph"
struct Foo {
    int a;
    uniform float b;
};

void update(varying Foo * uniform fp) {
    fp->a += 1;
    fp->b = 1;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    Foo f = { aFOO[programIndex], 5 };
    update(&f);
    RET[programIndex] = f.b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
