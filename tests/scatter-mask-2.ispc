#include "../test_static.isph"
uniform float a[programCount];

static void foo(int index) {
    #pragma ignore warning(perf)
    a[index] = 1;
}

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int index = aFOO[programIndex]-1; 
    if (index & 1)
        foo(index);
    RET[programIndex] = a[programIndex]; 
}

task void result(uniform float RET[]) {
    RET[programIndex] = (programIndex & 1) ? 1 : 0;
}
