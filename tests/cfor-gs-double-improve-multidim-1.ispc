
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex]; 
    uniform double udx[25][25];
    cfor (uniform int i = 0; i < 25; ++i)
        cfor (uniform int j = 0; j < 25; ++j)
            udx[i][j] = 10*i+j;

    int x = 1;
    RET[programIndex] = udx[x][programIndex+1];
}

export void result(uniform float RET[]) { RET[programIndex] = 11+programIndex; }
