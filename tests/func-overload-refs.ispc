#include "../test_static.isph"
float foo(float &a) { return 1; }
float foo(const float &a) { return 2; }

task void f_f(uniform float RET[], uniform float aFOO[]) {
    float x = 0;
    RET[programIndex] = foo(x); 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
