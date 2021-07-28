#include "../test_static.isph"
struct Foo {
    float x;
};

float bar(varying Foo f) { ++f.x; return f.x; }

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    varying Foo f = { a };
    bar(f);
    RET[programIndex] = f.x;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
