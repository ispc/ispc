export void f_f(uniform float RET[], uniform float aFOO[]) {
    RET[programIndex] = -1;
    int32 a = (programIndex & 1) ? 0xff : 0;
    if (programIndex & 1) {
        RET[programIndex] = exclusive_scan_and(a);
    }
}


export void result(uniform float RET[]) {
    if (programIndex & 1) {
        if (programIndex == 1)
            RET[programIndex] = -1; // 0xffffffff, made float
        else
            RET[programIndex] = 0xff;
    }
    else
        RET[programIndex] = -1;
}
