#include "../test_static.isph"
struct Foo {
    float x;
    float f;
};

task void f_fi(uniform float RET[], uniform float aFOO[], uniform int bFOO[]) {
    float a = aFOO[programIndex];
    int b = bFOO[programIndex];
    varying Foo myFoo = { a, b };
    RET[programIndex] = myFoo.x + myFoo.f;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 3 + 3 * programIndex;
}
