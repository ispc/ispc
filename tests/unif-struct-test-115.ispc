#include "../test_static.isph"
struct Foo {
    uniform float x;
    uniform float f;
};
task void f_fi(uniform float RET[], uniform float a[], uniform int b[]) {
    uniform struct Foo myFoo = { 1, 2 };
    RET[programIndex] = myFoo.x;
}


task void result(uniform float RET[4]) {
    RET[programIndex] = 1;
}
