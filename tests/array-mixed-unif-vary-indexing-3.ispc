
export uniform int width() { return programCount; }


export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    uniform float x[47][47];
    for (uniform int i = 0; i < 47; ++i)
        for (uniform int j = 0; j < 47; ++j)
            x[i][j] = 2+b-5;

    // all are 2 except (4,2) = 0, (4,...) = 1, (4,programCount-1)=2
    if (a == 3.)
        x[b-1][a-1] = 0;
    else
        x[b-1][a-1] = 1;
    RET[programIndex] = x[4][a];
}

export void result(uniform float RET[]) { 
    RET[programIndex] = 1;
    RET[1] = 0;
    RET[programCount-1] = 2;
}
