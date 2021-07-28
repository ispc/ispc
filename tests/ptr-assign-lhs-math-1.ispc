#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float a[programCount+1];
    a[programIndex] = aFOO[programIndex];
    a[programCount] = 1;

    uniform float * uniform ptr = a;
    *(ptr+1) = 0;
    RET[programIndex] = a[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
    if (programCount > 0)
        RET[1] = 0;
}
