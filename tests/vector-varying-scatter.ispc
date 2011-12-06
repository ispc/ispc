
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float aa = aFOO[programIndex]; 
    float<4> a = { -1, -2, -3, -4 };
    if (programIndex < 4)
        a[3-programIndex] = aa;
    RET[programIndex] = a[2];
}

export void result(uniform float RET[]) {
    RET[programIndex] = -3;
    RET[1] = 2;
}
