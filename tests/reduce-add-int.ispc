#include "../test_static.isph"
task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float v = aFOO[programIndex];
    uniform float m;
    int iv = (int)v;
/*CO    if (iv & 1)*/
        m = reduce_add(iv);
    RET[programIndex] = m;
}

task void result(uniform float RET[]) {
    uniform int x = 0;
    for (uniform int i = 1; i <= programCount; ++i)
        x += i;
    RET[programIndex] = x;
}
