#include "../test_static.isph"
struct Foo {
    uniform float a;
    uniform int8 b;
    uniform float c;
};

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = sizeof(Foo); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 12;
}
