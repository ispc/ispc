#include "../test_static.isph"
int64 zero = 0;
void *gptr;

task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform int a[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        a[i] = aFOO[i];
    
    int *ptr = (aFOO[0] == 1234) ? (int * varying)gptr : (a + programIndex);
    #pragma ignore warning(perf)
    int g = *ptr; 
    RET[programIndex] = g; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
