#include "../test_static.isph"
#define N 64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 uniform buf[N];

    foreach (i = 0 ... N)
        buf[i] = i;

    memmove(&buf[1], buf, (N-1)*sizeof(uniform int));
    RET[programIndex] = buf[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = max(0, programIndex-1);
}
