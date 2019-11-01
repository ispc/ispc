static float x[1][2];

export void f_f(uniform float RET[], uniform float aFOO[]) {
    float a = aFOO[programIndex];
    x[0][1] = a;
    RET[programIndex] = x[0][1];
}

export void result(uniform float RET[]) { RET[programIndex] = 1+programIndex; }
