#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

task void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    double v = aFOO[programIndex];
    uniform float m = 42;
    int iv = (int)v;
    if (iv & 1)
        m = reduce_add((double)iv);
    RET[programIndex] = m;
}

task void result(uniform float RET[]) {
    uniform int x = 0;
    for (uniform int i = 1; i <= programCount; i += 2)
        x += i;
    RET[programIndex] = x;
}
