#include "../test_static.isph"
bool foo(int *ptr) {
    return (ptr == NULL);
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a = 1;
    uniform int * uniform b = 0;
    RET[programIndex] = foo(0);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
