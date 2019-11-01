export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float udx[5][5];
    for (uniform int i = 0; i < 5; ++i)
        for (uniform int j = 0; j < 5; ++j)
            udx[i][j] = 10*i+j;

    int x = 1;
    RET[programIndex] = udx[x][(int)b-2];
}

export void result(uniform float RET[]) { RET[programIndex] = 13; } 
