// rule: skip on cpu=TGLLP

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[5][5];
    #pragma ignore warning
    cfor (uniform int i = 0; i < 5; ++i)
       #pragma ignore warning 
       cfor (uniform int j = 0; j < 5; ++j)
            udx[i][j] = 10*i+j;

    int x = 1;
    RET[programIndex] = udx[b-4][x];
}

export void result(uniform float RET[]) { RET[programIndex] = 11; }
