#include "../test_static.isph"
struct Foo { };

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform Foo f;
    RET[programIndex] = sizeof(f); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
