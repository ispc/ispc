export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 x[programCount];
    x[programIndex] = -1;
    int a = aFOO[programIndex]-1;
    if (programIndex < 3)
        x[a] = programIndex;
    RET[programIndex] = x[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = -1;
    RET[0] = 0;
    RET[1] = 1;
    RET[2] = 2;
}
