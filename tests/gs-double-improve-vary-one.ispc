// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[5] = { 1,2,3,4,5 };
    int x = 1;
    RET[programIndex] = udx[x];
}

export void result(uniform float RET[]) { RET[programIndex] = 2; }
