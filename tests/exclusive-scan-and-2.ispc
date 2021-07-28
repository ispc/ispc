#include "../test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int32 a = ~(1ul << programIndex);
    if ((programIndex < 32) && (programIndex & 1) == 0) {
        RET[programIndex] = exclusive_scan_and(a);
    }
}


task void result(uniform float RET[]) {
    RET[programIndex] = -1;
    if ((programIndex & 1) == 0 && programIndex > 0 && programIndex < 32) {
        int val = 0xffffffff;
        for (int i = 0; i < programIndex-1; i += 2)
            val &= ~(1ul<<i);
        RET[programIndex] = val;
    }
}
