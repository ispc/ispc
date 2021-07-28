#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    int i = (int)a;
    RET[programIndex] = min((int)-2, -1 * i);
    RET[programCount-1] = min(-2, -1 * (int)b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = - programIndex - 1;
    RET[0] = -2;
    RET[programCount - 1] = -5;
}
