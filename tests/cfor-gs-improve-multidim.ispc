
export uniform int width() { return programCount; }



export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform float udx[18][18];
    cfor (uniform int i = 0; i < 18; ++i)
        cfor (uniform int j = 0; j < 18; ++j)
            udx[i][j] = 100*i+j;

    int x = 1;
    RET[programIndex] = udx[x][x];
}

export void result(uniform float RET[]) { RET[programIndex] = 101; }
