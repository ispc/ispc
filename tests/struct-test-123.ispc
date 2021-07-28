#include "../test_static.isph"
struct Foo {
    float x;
};
void bar(varying Foo * uniform f) { ++((*f).x); }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    varying Foo f = {a};
    if (a == 1 || a == 4)
        bar(&f);
    RET[programIndex] = f.x;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
    RET[0] = 2;
    RET[3] = 5;
}
