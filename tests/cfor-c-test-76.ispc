export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    uniform float i;
    #pragma ignore warning
    cfor (i = 0; i < b; ++i) {
        if (i+2 == b) continue;
        ++a;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 5,6,7,8 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
