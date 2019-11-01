export void f_fu(uniform float RET[], uniform float xxFOO[], uniform float a) {
    float xx = xxFOO[programIndex];
    uniform float x[4][3];
    uniform int i, j;
    #pragma ignore warning
    cfor (i = 0; i < 4; ++i)
        #pragma ignore warning
        cfor (j = 0; j < 3; ++j)
            x[i][j] = a * i * j;
    RET[programIndex] = x[3][2];
}

export void result(uniform float RET[]) {
    RET[programIndex] = 30.000000;
}
