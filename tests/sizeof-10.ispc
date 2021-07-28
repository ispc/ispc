#include "../test_static.isph"
struct Foo {
    float a;
    uniform double b;
};

task void f_f(uniform float RET[], uniform float aFOO[]) {
    Foo array[20];
    RET[programIndex] = sizeof(array) / sizeof(array[0]);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 20;
}
