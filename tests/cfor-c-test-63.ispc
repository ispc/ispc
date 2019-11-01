export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i;
    cfor (i = 0; i < b; ++i) {
        if (i == 2) break;
        ++a;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 3, 4, 5, 6 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
