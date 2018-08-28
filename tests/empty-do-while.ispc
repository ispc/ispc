export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    do {} while (false);
    RET[programIndex] = (float)programIndex;
}

export void result(uniform float RET[]) {
    RET[programIndex] = (float)programIndex;
}
