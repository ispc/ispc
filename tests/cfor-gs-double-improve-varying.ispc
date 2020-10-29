// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    double udx[5];
    #pragma ignore warning
    cfor (uniform int i = 0; i < 5; ++i)
        udx[i] = i + a;

    int x = 1;
    RET[programIndex] = udx[x];
}

export void result(uniform float RET[]) { RET[programIndex] = 2+programIndex; }
