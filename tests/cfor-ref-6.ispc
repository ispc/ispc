#include "../test_static.isph"
void foo(float a[10]) {
    a[5] = 0;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float x[10];
    uniform int i;
    #pragma ignore warning
    cfor (i = 0; i < 10; ++i)
        x[i] = a*b;
    if (a >= 2)
        foo(x);
    RET[programIndex] = x[b] + x[9];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 5 * (programIndex+1);
    RET[0] = 10;
    RET[1] = 10;
}
