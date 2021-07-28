#include "../test_static.isph"
typedef int<3> int3;


task void f_fu(uniform float ret[], uniform float aa[], uniform float b) {
    float a = aa[programIndex];
    int3 v0 = { b, 2*b, 3*b };
    int3 v1 = { a, 2*a, 3*a };
    if (programIndex & 1)
        ret[programIndex] = (v0+v1).y;
    else
        ret[programIndex] = (v0+v1).z;
}

task void result(uniform float ret[]) {
    ret[programIndex] = (programIndex & 1) ? 10 + 2 * (programIndex+1) :
        15 + 3 * (programIndex+1);
}
