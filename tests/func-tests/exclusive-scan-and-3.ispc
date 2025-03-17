#include "test_static.isph"
task void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int8 a = (int8)~(1ul << (min(programIndex, 7)));
    if ((programIndex < 8) && (programIndex & 1) == 0) {
        RET[programIndex] = exclusive_scan_and(a);
    }
}


task void result(uniform float RET[]) {
    RET[programIndex] = -1;
    if ((programIndex & 1) == 0 && programIndex > 0 && programIndex < 8) {
        int8 val = -1; // All bits set for signed int8 (0xFF)
        for (int i = 0; i < programIndex-1; i += 2)
            val &= (int8)~(1ul << (min(i, 7)));
        RET[programIndex] = val;
    }
}
