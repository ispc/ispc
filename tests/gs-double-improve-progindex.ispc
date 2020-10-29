// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[programCount];
    for (uniform int i = 0; i < programCount; ++i)
        udx[i] = i+1;
    int x = programIndex;
    RET[programIndex] = udx[x];
}

export void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
