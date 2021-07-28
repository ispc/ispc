#include "../test_static.isph"
void inc(varying int * uniform v) {
    ++*v;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int a = aFOO[programIndex];
    varying int * uniform b = &a;
    void * uniform vp = b;
    varying int * uniform c = (varying int * uniform)vp;
    RET[programIndex] = *c;
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
