export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        ++a;
        cfor (j = 0; j < b; ++j) {
            if (a == 3.) break;
            ++a;
        }
        ++a;
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[]) {
    uniform float ret[4] = { 32, 32, 38, 39 };
    #pragma ignore warning(perf)
    RET[programIndex] = ret[programIndex & 0x3];
}
