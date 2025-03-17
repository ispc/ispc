#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int16 a = (int16)~(1ul << (min(programIndex, 15)));
    if ((programIndex < 16) && (programIndex & 1) == 0) {
        RET[programIndex] = exclusive_scan_and(a);
    }
}


task void result(uniform float RET[]) {
    RET[programIndex] = -1;
    if ((programIndex & 1) == 0 && programIndex > 0 && programIndex < 16) {
        int16 val = -1; // All bits set for signed int16 (0xFFFF)
        for (int i = 0; i < programIndex-1; i += 2)
            val &= (int16)~(1ul << (min(i, 15)));
        RET[programIndex] = val;
    }
}
