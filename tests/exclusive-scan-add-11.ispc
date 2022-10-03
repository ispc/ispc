#include "../test_static.isph"
// rule: skip on arch=x86
// rule: skip on arch=x86-64
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    float16 a = aFOO[programIndex];
    if (programIndex & 1) {
        RET[programIndex] = exclusive_scan_add(a);
    }
}

int es(int v) {
    uniform int vv[programCount];
    vv[programIndex] = v;

    uniform int r[programCount];
    r[0] = 0;
    for (uniform int i = 1; i < programCount; ++i)
        r[i] = r[i-1] + vv[i-1];
    return r[programIndex];
}


task void result(uniform float RET[]) {
    RET[programIndex] = es((programIndex & 1) ? (programIndex+1) : 0);
    if ((programIndex & 1) == 0)
        RET[programIndex] = -1;
}
