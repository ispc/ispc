#include "../test_static.isph"
struct Foo {
    float x;
    float f;
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    struct Foo myFoo = {a,a};
    RET[programIndex] = myFoo.x;
}


task void result(uniform float RET[4]) {
    RET[programIndex] = 1+programIndex;
}
