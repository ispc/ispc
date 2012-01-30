
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = programIndex;
    RET[programIndex] = extract(a, min(programCount-1, 3)); 
}

export void result(uniform float RET[]) {
    RET[programIndex] = (programCount == 1) ? 0 : 3;
}
