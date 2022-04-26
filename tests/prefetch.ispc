#include "../test_static.isph"

task void f_f(uniform float RET[], uniform float aFOO[]) {
    prefetch_l1(aFOO);
    prefetch_l2(aFOO);
    prefetch_l3(aFOO);
    prefetch_nt(aFOO);
    float a = aFOO[programIndex];
    float b = 0.; b = a;
    RET[programIndex] = a+b;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 2 + 2*programIndex;
}
