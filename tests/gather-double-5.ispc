#include "../test_static.isph"
// rule: skip on cpu=tgllp
// rule: skip on cpu=dg2

int zero = 0;
void *gptr;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform double a[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = aFOO[i];

    double *ptr = (aFOO[0] == 1234) ? (double * varying)gptr : (a + programIndex);
    #pragma ignore warning(perf)
    int g = *ptr;
    RET[programIndex] = g; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
