
export uniform int width() { return programCount; }

export void f_f(uniform float RET[], uniform float aFOO[]) {
    double a = aFOO[programIndex]; 
    RET[programIndex] = abs(-a);
}

export void result(uniform float RET[]) {
    RET[programIndex] = 1 + programIndex;
}
