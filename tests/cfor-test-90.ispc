#include "../test_static.isph"
float foo(uniform float x[3]) {
    return x[1];
}
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[3];
    uniform int i;
    #pragma ignore warning
    cfor (i = 0; i < 3; ++i) x[i] = b;
    RET[programIndex] = foo(x);
}


task void result(uniform float RET[]) {
    RET[programIndex] = 5.000000;
}
