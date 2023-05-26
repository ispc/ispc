#include "../test_static.isph"
// rule: skip on arch=xe64

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 * uniform buf = uniform new int32[1024];

    foreach (i = 0 ... 1024)
        buf[i] = i;

    memmove(&buf[1], buf, (1024-1)*sizeof(uniform int));
    RET[programIndex] = buf[programIndex];
}

task void result(uniform float RET[]) {
    RET[programIndex] = max(0, programIndex-1);
}
