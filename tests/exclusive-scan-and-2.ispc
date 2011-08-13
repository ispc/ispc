
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int32 a = ~(1 << programIndex);
    if ((programIndex & 1) == 0) {
        RET[programIndex] = exclusive_scan_and(a);
    }
}


export void result(uniform float RET[]) {
    RET[programIndex] = -1;
    if ((programIndex & 1) == 0 && programIndex > 0) {
        int val = 0xffffffff;
        for (int i = 0; i < programIndex-1; i += 2)
            val &= ~(1<<i);
        RET[programIndex] = val;
    }
}
