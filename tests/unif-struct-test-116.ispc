#include "../test_static.isph"
struct Foo {
    uniform float x;
    uniform float f;
};

task void f_fi(uniform float RET[], uniform float a[], uniform int b[]) {
    uniform struct Foo myFoo = {2,2};
    RET[programIndex] = myFoo.x + myFoo.f;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 4;
}
