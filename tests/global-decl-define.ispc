#include "../test_static.isph"
extern int foo;
int foo = 1;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = foo; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
