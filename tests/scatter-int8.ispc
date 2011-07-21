export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int8 x[programCount];
    x[programIndex] = 0;
    int a = aFOO[programIndex]-1;
    x[a] = programIndex;
    RET[programIndex] = x[programIndex];
}

export void result(uniform float RET[]) {
    RET[programIndex] = programIndex;
}
