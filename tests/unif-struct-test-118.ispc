#include "../test_static.isph"
struct Foo {
    float x;
    float f;
};
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    struct Foo myFoo[3] = { { a,a}, {a,a}, {a,a} };
    struct Foo barFoo;
    barFoo = myFoo[1];
    RET[programIndex] = barFoo.x;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
