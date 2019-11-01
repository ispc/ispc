export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        if (a == b) continue;
        ++a;
        if (a == 2) break;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 2, 5, 5, 5 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
