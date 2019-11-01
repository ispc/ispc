float foo(uniform float x[3]) {
    return x[1];
}
export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    uniform float x[3][3];
    uniform int i, j;
    for (i = 0; i < 3; ++i) 
        for (j = 0; j < 3; ++j)
            x[i][j] = 3 + i*j;
    RET[programIndex] = foo(x[1]);
}


export void result(uniform float RET[]) {
    RET[programIndex] = 4.000000;
}
