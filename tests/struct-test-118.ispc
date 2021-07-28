#include "../test_static.isph"
struct Foo {
    float x;
    float f;
};

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    Foo myFoo[3] = { { -1, -2 }, {a, -3}, {-4, -5} };
    int i = aFOO[0];
    Foo barFoo = myFoo[i];
//CO    print("% %\n", myFoo[i].x, barFoo.x);
    RET[programIndex] = barFoo.x;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
