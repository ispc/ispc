#include "../test_static.isph"
float foo(uniform float * uniform a) {
    *a = 0;
}

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    foo(&b);
    RET[programIndex] = b;
}

task void result(uniform float RET[]) { RET[programIndex] = 0; }
