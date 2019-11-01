export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform int i, j;
    float x[5][2];
    for (i = 0; i < b; ++i)
        for (j = 0; j < 2; ++j)
            x[i][j] = i+j;
    RET[programIndex] = x[b-4][b-4];
}


export void result(uniform float RET[4]) {
    RET[programIndex] = 2.;
}
