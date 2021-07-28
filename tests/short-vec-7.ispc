#include "../test_static.isph"
typedef int<4> int4;


void incXY(varying int4 * uniform v) {
    ++(*v).x;
    ++(*v).y;
}

task void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    int4 v0 = { b, 2*b, 3*b, 2 };
    incXY(&v0);
    if (programIndex & 1) incXY(&v0);
    ret[programIndex] = v0.x + v0.y;
}

task void result(uniform float ret[]) {
    ret[programIndex] = 17;
    if (programIndex & 1) ret[programIndex] += 2;
}
