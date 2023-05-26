#include "../test_static.isph"
// rule: skip on arch=xe64

task void f_f(uniform float RET[], uniform float aFOO[]) {
    int32 * uniform buf = uniform new int32[1024];
    buf[0] = 0;
    memset(buf+1, 0x7f, 1024*sizeof(uniform int32));

    int v = buf[programIndex];
    RET[programIndex] = (v == 0x7f7f7f7f);
}

task void result(uniform float RET[]) {
    RET[programIndex] = 1;
    RET[0] = 0;
}
