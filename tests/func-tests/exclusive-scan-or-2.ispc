#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int8 a = (int8)(1ul << (min(programIndex, 6)));
    RET[programIndex] = exclusive_scan_or(a);
}


task void result(uniform float RET[]) {
    #pragma ignore warning(perf)
    RET[programIndex] = (int8)((1ul << (min(programIndex, 7))) - 1);
}
