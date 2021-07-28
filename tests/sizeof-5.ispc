#include "../test_static.isph"
struct Foo {
    uniform int16 a;
    uniform unsigned int16 c;
};

task void f_f(uniform float RET[], uniform float aFOO[]) {
    Foo a[10];
    RET[programIndex] = sizeof(a);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 40;
}
