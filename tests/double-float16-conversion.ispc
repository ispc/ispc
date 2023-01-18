#include "../test_static.isph"

task void f_du(uniform float RET[], uniform double aFOO[], uniform double b) {
    double a = aFOO[programIndex];
    float16 X = a;
    float16 Y = b;
    float16 res = X * Y;
    if (res == 5.0f16 * (programIndex + 1)) {
        RET[programIndex] = 0;
    } else {
        RET[programIndex] = 1;
    }
}

task void result(uniform float RET[]) { RET[programIndex] = 0;}
