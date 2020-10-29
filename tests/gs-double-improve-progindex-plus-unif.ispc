// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    int x = -5 + programIndex + (int)b;
    #pragma ignore warning(perf)
    RET[programIndex] = udx[x % 16];
}

export void result(uniform float RET[]) { RET[programIndex] = 1+(programIndex%16); }
