#include "../test_static.isph"
int func(const int &a) { return a+1; }
int bar() { return 0; }

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = func(bar());
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
