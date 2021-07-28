#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    int i = (int)a;
    RET[programIndex] = max((int)-2, -1 * i);
    RET[programCount-1] = max(-2, -1 * (int)b);
}

task void result(uniform float RET[]) {
    RET[programIndex] = -2;
    RET[0] = -1;
}
