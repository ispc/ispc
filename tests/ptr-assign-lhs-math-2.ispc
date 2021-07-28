#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    uniform float a[programCount];
    a[programIndex] = aFOO[programIndex];

    uniform float * varying ptr = a;
    *(ptr+programIndex) = 0;
    RET[programIndex] = a[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 0;
}
