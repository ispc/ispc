#include "test_static.isph"

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m;
    int8 iv = (int)v;
    if (iv & 1)
        m = reduce_add((iv % 11) - 5);
    RET[programIndex] = m;
}

task void result(uniform float RET[]) {
    uniform int8 x = 0;
    for (uniform int i = 1; i <= programCount; i += 2) {
        x += (i % 11) - 5;
    }
    RET[programIndex] = x;
}
