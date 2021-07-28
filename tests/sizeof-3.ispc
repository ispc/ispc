#include "../test_static.isph"
struct Foo {
    float a;
    int8 b;
};

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = sizeof(Foo); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 8*programCount;
}
