#include "../test_static.isph"
// rule: skip on arch=xe64

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 * varying buf = varying new int32[1024*(programIndex+1)];

    if (programIndex & 1) {
        memset(buf, 0xff, 1024*(programIndex+1)*sizeof(uniform int32));
    }
    else {
        memset(buf, 0x01, 1024*(programIndex+1)*sizeof(uniform int32));
    }

    #pragma ignore warning(perf)
    int v = buf[0];
    int expected = (programIndex & 1) ? 0xffffffff : 0x01010101;
    RET[programIndex] = (v == expected);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
}
