#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int16 a = (int16)(1ul << (min(programIndex, 14)));
    RET[programIndex] = exclusive_scan_or(a);
}


task void result(uniform float RET[]) {
    #pragma ignore warning(perf)
    RET[programIndex] = (int16)((1ul << (min(programIndex, 15))) - 1);
}
