#include "../test_static.isph"
struct Foo {
    float x;
    float f;
    int i[3];
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    varying Foo myFoo[3] = { { b, b, {b, b, b} }, 
                             { b, b, {b, b, b} }, 
                             { b, b, {b, b, b} } };
    Foo barFoo;
    barFoo = myFoo[0];
    RET[programIndex] = barFoo.x + myFoo[1].i[2];
}


task void result(uniform float RET[]) {
    RET[programIndex] = 10;
}
