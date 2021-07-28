#include "../test_static.isph"
typedef float (*func)();

float foo();
float bar();

struct X { func f, g; };

static uniform X x = { foo, &bar };

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = x.f() + x.g();
}

task void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}

float foo() {
    return 2 * programIndex;
}

float bar() {
    return -programIndex;
}
