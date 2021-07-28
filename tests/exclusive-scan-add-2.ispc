#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = exclusive_scan_add(aFOO[programIndex]);
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
    RET[programIndex] = es(programIndex+1); 
}
