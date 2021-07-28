#include "../test_static.isph"
static uniform float x = 2;


task void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    if (b > 0)
        x = 0;
    ret[programIndex] = x;
}

task void result(uniform float ret[]) {
    ret[programIndex] = 0;
}
