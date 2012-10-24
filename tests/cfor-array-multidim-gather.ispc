
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[4][3];
    uniform int i, j;
    cfor (i = 0; i < 4; ++i)
        cfor (j = 0; j < 3; ++j)
            x[i][j] = 3*i+j;

    a = clamp(a-2., 0., 2.);
    RET[programIndex] = x[a][a];
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 8;
    RET[0] = RET[1] = 0;
    RET[2] = 4;
}
