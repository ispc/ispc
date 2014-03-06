
export uniform int width() { return programCount; }



export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    float a = aFOO[programIndex];
    int i = (int)a;
    RET[programIndex] = max((int)-2, -1 * i);
    RET[width() - 1] = max(-2, -1 * (int)b);
}

export void result(uniform float RET[]) {
    RET[programIndex] = -2;
    RET[0] = -1;
}

