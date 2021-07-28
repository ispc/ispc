#include "../test_static.isph"
struct Foo {
    varying int i;
    varying float f;
};
task void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform struct Foo myFoo;
    myFoo.f = a;
    RET[programIndex] = myFoo.f + a;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
