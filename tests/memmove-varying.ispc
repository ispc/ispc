#include "../test_static.isph"
// rule: skip on arch=xe32
// rule: skip on arch=xe64

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 *buf = new int32[1024];

    for (uniform int i = 0; i < 1024; ++i) {
        #pragma ignore warning(perf)
        buf[i] = programIndex * 10000 + i;
    }

    if (programIndex == 2)
        memmove(buf, buf+programCount/2, programCount*sizeof(uniform int));

    #pragma ignore warning(perf)
    RET[programIndex] = buf[0];
}

task void result(uniform float RET[]) {
    RET[programIndex] = 10000 * programIndex;
    RET[2] = 10000 * 2 + programCount/2;
}
