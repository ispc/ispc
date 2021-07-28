#include "../test_static.isph"
struct Foo {
    varying float x;
};
float bar(uniform struct Foo * uniform f) { ++((*f).x); return (*f).x; }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform struct Foo f = {a};
    bar(&f);
    RET[programIndex] = f.x;
}


task void result(uniform float RET[4]) {
    RET[programIndex] = 2 + programIndex;
}
