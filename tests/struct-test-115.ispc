#include "../test_static.isph"
struct Foo {
    varying float x;
    uniform float f;
};

task void f_fi(uniform float RET[], uniform float aFOO[], uniform int b[]) {
    float a = aFOO[programIndex];
    varying Foo myFoo = { a, 2 };
    RET[programIndex] = myFoo.x;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
