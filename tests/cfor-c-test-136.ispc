export void f_fu(uniform float RET[4], uniform float aFOO[4], uniform float b) {
    #pragma ignore warning(perf)
    float a = aFOO[programIndex&0x3];
    float i, j;
    cfor (i = 0; i < b; ++i) {
        cif (a == 1.) break;
        cfor (j = 0; j < b; ++j) {
            cif (a == 3.) break;
            ++a;
        }
    }
    RET[programIndex] = a;
}


export void result(uniform float RET[4]) {
    uniform float ret[4] = { 1,3,3,29 };
    #pragma ignore warning(perf)    
    RET[programIndex] = ret[programIndex & 0x3];
}
