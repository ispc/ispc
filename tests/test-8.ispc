#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    float r = a;
    if (b == 5.) r = b;
    RET[programIndex] = r;
}


task void result(uniform float RET[]) {
    RET[programIndex] = 5;
}
