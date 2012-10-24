export uniform int width() { return programCount; }

export void f_fu(uniform float RET[], uniform float aFOO[], uniform float b) {
    uniform int16 x[programCount];
    x[programIndex] = aFOO[programIndex];
    unsigned int16 v = x[programIndex];
    RET[programIndex] = v;
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1+programIndex;
}
